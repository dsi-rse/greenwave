"""
Preprocess GreenWave's data
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

YCOL, TCOL, IDCOL = "lbs_ft", "day_of_season", "farm_season_id"
SAMPLE_EVENT, HARVEST_EVENT, OUTPLANT_EVENT = "sample", "harvest", "outplant"
GROUP = ["Farm Name", "Season", "Species"]
# older exports (e.g. 20260416) call the farm column "Anon Farm"
FARM_ALIASES = {"Anon Farm": "Farm Name"}

MODEL_SPECIES = "sugar_kelp"
MIN_FARMS_PER_REGION = 3
#leeway on "total line harvested must not exceed what was outplanted"
LINE_TOLERANCE = 0.0

# the per-event flags every model run applies
DEFAULT_FILTERS = ("flag_model_species", "flag_has_outplant_date",
                   "flag_harvest_within_available", "flag_after_seed")


def day_of_season(ts) -> int:
    """
    Days since the most recent Oct 1.

    Outplanting happens in late fall or early winter, so the anchor sits before any season starts.
    Crop age (days since outplant) was not used: it removes planting-date variance across farms, it
    is ambiguous when a farm-season has several outplant dates (54 of 155 do), and it drops the
    environmental information in the calendar date. GreenWave noted planting date as important, with
    earlier outplanting tending to a bigger harvest.

    Args:
        ts (str | datetime | pd.Timestamp): the log date to convert.

    Returns:
        int: days from the preceding Oct 1.
    """
    ts = pd.Timestamp(ts)
    oct1 = pd.Timestamp(year=ts.year if ts.month >= 10 else ts.year - 1,
                            month=10, day=1)

    return (ts - oct1).days


def build_events(path: str) -> pd.DataFrame:
    """
    Build the events table from the raw GreenWave Logs sheet.

    One row per logged outplant, sample, or harvest, grouped by farm, season, and species. Each
    event carries its farm-season's seed date (first outplant), total outplanted line, lost line,
    harvested line, and the flags that say whether the farm-season is usable.

    Samples carry `line_ft = 1.0` because a sample is a one-foot pull, so `lbs_ft` and `weight_lbs`
    match. Harvests keep both separately, which is what lets a lbs/ft forecast become pounds.

    Flags, none of them applied here (see `load_events`):
        flag_model_species: species is `MODEL_SPECIES` (sugar kelp).
        flag_single_outplant: the farm-season had exactly one outplant date.
        flag_has_harvest: at least one harvest was logged.
        flag_harvest_within_available: harvested line is within outplanted minus lost, with
            `LINE_TOLERANCE` leeway.
        flag_region_ok: the state has at least `MIN_FARMS_PER_REGION` farms.
        flag_after_seed: the event falls after the first outplant. Outplants pass by construction.
        flag_has_outplant_date: a seed date exists.

    Args:
        path (str): path to the GreenWave export .xlsx. Reads the "Logs" sheet.

    Returns:
        pd.DataFrame: one row per event, unfiltered, with the flag columns above.
    """
    logs = pd.read_excel(path, sheet_name="Logs").rename(columns=FARM_ALIASES)
    logs["Log Date"] = pd.to_datetime(logs["Log Date"]).dt.normalize()
    logs.dropna(subset=["Log Date"], inplace=True)

    outplant = logs[logs["Log Type"] == "outplanting"].copy()
    samples = logs[logs["Log Type"] == "sample"].copy()
    harvest = logs[logs["Log Type"] == "harvest"].copy()

    loss_types = [t for t in logs["Log Type"].dropna().unique()
                      if "loss" in str(t).lower()]
    loss = logs[logs["Log Type"].isin(loss_types)].copy()

    # get the total line outplanted, lost, and harvested in a farm-season
    # not a running total but the final amounts
    grp_outplant = (outplant.groupby(GROUP)
        .agg(seed_date=("Log Date", "min"),
             last_outplant_date=("Log Date", "max"),
             n_outplant_dates=("Log Date", "nunique"),
             outplant_line_ft=("Line Length", "sum")).reset_index())

    grp_harvest = (harvest.dropna(subset=["Weight"]).groupby(GROUP)
        .agg(true_biomass_lbs=("Weight", "sum"),
             harvest_line_ft=("Line Length", "sum"),
             n_harvests=("Weight", "size")).reset_index())

    grp_loss = (loss.groupby(GROUP).agg(loss_line_ft=("Line Length", "sum"))
                .reset_index()
                if len(loss) else pd.DataFrame(columns=GROUP + ["loss_line_ft"]))

    grp_outplant["outplant_spread_days"] = (grp_outplant["last_outplant_date"]
                                            - grp_outplant["seed_date"])
    group_biomass = (grp_outplant.merge(grp_harvest, on=GROUP, how="left")
                          .merge(grp_loss, on=GROUP, how="left"))
    group_biomass["loss_line_ft"] = group_biomass["loss_line_ft"].fillna(0.0)
    group_biomass["available_line_ft"] = (group_biomass["outplant_line_ft"]
                                          - group_biomass["loss_line_ft"])
    group_biomass["flag_single_outplant"] = group_biomass["n_outplant_dates"] == 1
    group_biomass["n_harvests"] = group_biomass["n_harvests"].fillna(0.0)
    group_biomass["flag_has_harvest"] = group_biomass["n_harvests"] > 0
    group_biomass["flag_harvest_within_available"] = ~group_biomass["flag_has_harvest"] | (
            group_biomass["harvest_line_ft"]
            <= group_biomass["available_line_ft"] * (1 + LINE_TOLERANCE))

    # add the state (region) per farm group
    farm_state = logs[["Farm Name", "State"]].dropna().drop_duplicates("Farm Name")
    group_biomass = group_biomass.merge(farm_state, on="Farm Name", how="left")
    sugar_kelp_groups = group_biomass[group_biomass["Species"] == MODEL_SPECIES]
    farms_per_state = sugar_kelp_groups.groupby("State")["Farm Name"].nunique()
    valid_states = farms_per_state[farms_per_state >= MIN_FARMS_PER_REGION].index
    group_biomass["flag_region_ok"] = group_biomass["State"].isin(valid_states)

    # add observation events
    outplant_ev = (outplant[GROUP + ["Log Date"]].assign(lbs_ft=0.0, event="outplant"))
    sample_ev = (samples.dropna(subset=["Weight"])[GROUP + ["Log Date", "Weight"]]
                .rename(columns={"Weight": "lbs_ft"})
                .assign(event="sample", line_ft=1.0))

    sample_ev["weight_lbs"] = sample_ev["lbs_ft"]  # samples are per 1 ft
    harvest_ev = harvest.dropna(subset=["Weight", "Line Length"]).copy()
    harvest_ev = harvest_ev[harvest_ev["Line Length"] > 0] # something had to be harvested
    harvest_ev["lbs_ft"] = harvest_ev["Weight"] / harvest_ev["Line Length"]
    # keep the raw pieces: line_ft turns a lbs/ft forecast back into pounds,
    # weight_lbs is the per-event biomass ground truth
    harvest_ev["line_ft"] = harvest_ev["Line Length"]
    harvest_ev["weight_lbs"] = harvest_ev["Weight"]
    harvest_ev = harvest_ev[GROUP + ["Log Date", "lbs_ft", "line_ft",
                                     "weight_lbs"]].assign(event="harvest")
    events = pd.concat([outplant_ev, sample_ev, harvest_ev], ignore_index=True)
    events["is_outplant"] = events["event"] == "outplant"

    # add its farm-season biomass information to each event
    events = events.merge(group_biomass, on=GROUP, how="left")
    # filter only to sugar kelp and fill in rest of flags + day of season
    events["flag_model_species"] = events["Species"] == MODEL_SPECIES
    events["flag_has_outplant_date"] = events["seed_date"].notna()

    for c in ["flag_single_outplant", "flag_has_harvest",
              "flag_harvest_within_available", "flag_region_ok"]:
        events[c] = events[c].astype("boolean").fillna(False).astype(bool)

    events["day_of_season"] = events["Log Date"].apply(day_of_season)
    events["flag_after_seed"] = events["is_outplant"] | (
            events["Log Date"] > events["seed_date"])
    events["farm_season_id"] = (
            events["Farm Name"].str.replace("Farm_", "F") + "_"
            + events["Season"].str.replace("Season ", "").str.replace("/", ""))

    return events



def load_events(xlsx):
    """
    Events table after the four filters every model run applies.

    Applies `DEFAULT_FILTERS` and adds `season_key` so seasons sort in time order. Prefer
    `load_dataset`, which also records where the data came from.

    Args:
        xlsx (str): path to the GreenWave export .xlsx.

    Returns:
        pd.DataFrame: filtered events with `season_key` added.
    """
    events = build_events(xlsx)

    for flag in DEFAULT_FILTERS:
        events = events[events[flag]]

    events = events.copy()
    events["season_key"] = events["Season"].map(season_key)

    return events.reset_index(drop=True)


def load_line_events(xlsx, events):
    """
    Every event that changes how much line is in the water, per farm-season.

    Outplanting adds line, line loss and harvest remove it. Harvests with no weight logged are
    included: no yield was recorded, but the line still came out of the water.

    Args:
        xlsx (str): path to the GreenWave export .xlsx.
        events (pd.DataFrame): a loaded events table, used to map farm/season/species to
            `farm_season_id` so the two agree.

    Returns:
        pd.DataFrame: `farm_season_id`, `day`, `Log Type`, `delta_ft` (signed feet).
    """
    logs = pd.read_excel(xlsx, sheet_name="Logs").rename(columns=FARM_ALIASES)
    logs["Log Date"] = pd.to_datetime(logs["Log Date"]).dt.normalize()
    logs = logs.dropna(subset=["Log Date"])
    sign = {"outplanting": 1.0, "line_loss": -1.0, "harvest": -1.0}
    line = logs[logs["Log Type"].isin(sign)].copy()
    line["day"] = line["Log Date"].apply(day_of_season)
    line["delta_ft"] = line["Log Type"].map(sign) * line["Line Length"].fillna(0.0)
    keys = events.drop_duplicates("farm_season_id")[GROUP + ["farm_season_id"]]
    line = line.merge(keys, on=GROUP)

    return line[["farm_season_id", "day", "Log Type", "delta_ft"]]


def line_in_water(line_events, fsid, day):
    """
    Feet of line in the water at the start of `day`.

    Sums the signed deltas from days strictly before `day`, so a harvest's own footage is not
    counted as still in the water when forecasting it.

    Args:
        line_events (pd.DataFrame): output of `load_line_events`.
        fsid (str): farm-season id, e.g. "F2_2526".
        day (float): day of season to evaluate at.

    Returns:
        float: net feet of line. Negative means the logs record more line out than in.
    """
    g = line_events[(line_events["farm_season_id"] == fsid) & (line_events["day"] < day)]

    return float(g["delta_ft"].sum())


def season_obs(events, fsid, include_outplant=False):
    """
    One farm-season's observations as a frame, sorted by day.

    The frame version of `season_events`, used by the hierarchy's stage-1 fit. `kind` is "sample" or
    "harvest". Outplants are excluded by default for the reasons in `season_events`.

    Args:
        events (pd.DataFrame): events table with outplants, samples, and harvests.
        fsid (str): farm-season id.
        include_outplant (bool): keep outplants as 0 lbs/ft sample observations. Defaults to False.

    Returns:
        pd.DataFrame: `day`, `y`, `kind`, `is_outplant`, `line_ft`, `weight_lbs`, sorted by day.
    """
    g = events[events["farm_season_id"] == fsid]

    if not include_outplant:
        g = g[g["event"] != "outplant"]

    obs = pd.DataFrame({
        "day": g["day_of_season"].astype(float),
        "y": g["lbs_ft"].astype(float),
        "kind": np.where(g["event"] == "harvest", "harvest", "sample"),
        "is_outplant": g["event"] == "outplant",
        "line_ft": g["line_ft"].astype(float),
        "weight_lbs": g["weight_lbs"].astype(float),
    })

    return obs.dropna(subset=["y"]).sort_values("day").reset_index(drop=True)



def season_key(season):
    """
    Season label to a sortable start year.

    Seasons arrive as strings ("Season 23/24") which sort lexically, not in time order. This is what
    lets `history` ask for strictly earlier seasons.

    Args:
        season (str): a season label, e.g. "Season 23/24".

    Returns:
        int: the starting year, e.g. 2023.
    """
    digits = "".join(ch for ch in str(season) if ch.isdigit())

    return int(digits[:2]) + 2000


def history(events, target_season):
    """
    All farm-seasons from seasons strictly before the target's.

    Args:
        events (pd.DataFrame): events table with `season_key`.
        target_season (str): the season being forecast, e.g. "Season 25/26".

    Returns:
        pd.DataFrame: events from earlier seasons only.
    """
    hist = events[events["season_key"] < season_key(target_season)]
    assert (hist["season_key"] < season_key(target_season)).all()

    return hist


def season_events(df, fsid):
    """
    One farm-season's observations as tuples in time order. What the posterior conditions on.

    Samples and harvests only, so every `n_obs` count in the reports counts samples and harvests and
    nothing else. Outplants still get a forecast in `evaluate.season_report`, from the posterior
    conditioned on whatever was logged earlier; that is a prediction at an outplant date, not a fit
    to one.

    Args:
        df (pd.DataFrame): events table with outplants, samples, and harvests.
        fsid (str): farm-season id.

    Returns:
        list[tuple]: `(day, lbs_ft, event, line_ft, weight_lbs)` sorted by day. Slice it to
            condition a fit on part of the season.
    """
    g = df[(df[IDCOL] == fsid) & df[YCOL].notna()
           & df["event"].isin([SAMPLE_EVENT, HARVEST_EVENT])]
    ev = [(float(r[TCOL]), float(r[YCOL]), r["event"],
           float(r.get("line_ft", np.nan)), float(r.get("weight_lbs", np.nan)))
          for _, r in g.iterrows()]

    return sorted(ev, key=lambda e: e[0])


def split(seen):
    """
    Sample and harvest observations as four arrays.

    The two channels enter the likelihood separately, samples against `f(t)` and harvests against
    `c * f(t)` with their own noise scale, so they are split before being handed to the model.

    Args:
        seen (list[tuple]): observations from `season_events`, usually an earlier prefix.

    Returns:
        tuple: `(ts, ys, th, yh)` -- sample days, sample yields, harvest days, harvest yields. Any
            of them may be empty.
    """
    ts = np.array([d for d, v, ch, *_ in seen if ch == SAMPLE_EVENT])
    ys = np.array([v for d, v, ch, *_ in seen if ch == SAMPLE_EVENT])
    th = np.array([d for d, v, ch, *_ in seen if ch == HARVEST_EVENT])
    yh = np.array([v for d, v, ch, *_ in seen if ch == HARVEST_EVENT])

    return ts, ys, th, yh


def data_fingerprint(seen):
    """
    Exact hash of the observations a posterior was conditioned on.

    Part of the posterior cache key.

    Args:
        seen (list[tuple]): the observations, from `season_events`.

    Returns:
        str: first 12 hex characters of the SHA-1 of `(day, lbs_ft, event)` per observation.
    """
    payload = repr([(round(float(d), 6), round(float(v), 6), str(k))
                    for d, v, k, *_ in seen])

    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_dataset(xlsx, filters=DEFAULT_FILTERS, verbose=True):
    """
    The events table plus a record of where it came from.

    The entry point for every notebook. `meta` is what lets a report state which extract and which
    filters produced it, so two runs can be compared without wondering whether the inputs matched.


    Args:
        xlsx (str): path to the GreenWave export .xlsx.
        filters (tuple[str, ...]): `flag_*` columns to require. Defaults to `DEFAULT_FILTERS`. Each
            is applied in order and its survivors recorded, so it is visible what dropped what.
        verbose (bool): print the row counts and fingerprints. Defaults to True.

    Returns:
        tuple[pd.DataFrame, dict]: the events, and `meta` with the source, file hash, filters, row
            and farm-season counts, per-filter survivors, and an 8-char `fingerprint`.
    """
    path = Path(xlsx).expanduser()
    file_hash = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    events = load_events(str(path))
    kept = {}

    for f in filters:
        if f in events.columns:
            events = events[events[f]]
            kept[f] = (len(events), events.groupby("farm_season_id").ngroups)

    events = events.reset_index(drop=True)
    meta = {"source": str(path), "file_sha1": file_hash,
            "filters": tuple(filters), "n_events": len(events),
            "n_farm_seasons": events["farm_season_id"].nunique(),
            "n_farms": events["Farm Name"].nunique(),
            "seasons": sorted(events["Season"].unique()),
            "after_each_filter": kept}

    meta["fingerprint"] = hashlib.sha1(
        repr((file_hash, tuple(filters), meta["n_events"])).encode()).hexdigest()[:8]

    if verbose:
        print(f"{meta['n_events']:,} events | {meta['n_farm_seasons']} farm-seasons | "
              f"{meta['n_farms']} farms | {len(meta['seasons'])} seasons")
        print(f"source {path.name} (sha1 {file_hash}) | data fingerprint {meta['fingerprint']}")

    return events, meta
