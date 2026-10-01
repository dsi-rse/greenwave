"""
Evaluation: predict and compare to what actually happened

Everything here needs both a forecast and the truth, which is what separates it from `model` 
(forecast only) and `plots` (plotting only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import (harvest_intervals, joint_mode_prediction, naive_lbs_ft,
                    point_estimate, sample_interval)
from .preprocess import HARVEST_EVENT, IDCOL, OUTPLANT_EVENT, SAMPLE_EVENT, TCOL, season_events

LABEL_NAIVE = "GreenWave today (last observation)"
LABEL_MODEL = "Growth model"



#: column groups, for `view`
REPORT_GROUPS = {
    "keys":     ["farm_season_id", "date", "day", "event"],
    "context":  ["n_samples_seen", "n_harvests_seen", "gap_days"],
    "truth":    ["actual_lbs_ft", "actual_lbs", "line_harvested_ft", "line_in_water_ft"],
    "before":   ["pred_sample_lbs_ft", "sample_yield_lo95", "sample_yield_hi95",
                 "pred_harvest_lbs_ft",
                 "harvest_yield_lo95", "harvest_yield_hi95",
                 "harvest_reading_lo95", "harvest_reading_hi95"],
    "after":    ["pred_sample_lbs_ft_after",
                 "sample_yield_lo95_after", "sample_yield_hi95_after",
                 "pred_harvest_lbs_ft_after",
                 "harvest_yield_lo95_after", "harvest_yield_hi95_after",
                 "harvest_reading_lo95_after", "harvest_reading_hi95_after"],
    "baseline": ["naive_lbs_ft"],
    "verdict":  ["in95_yield", "in95_reading"],
    "pounds":   ["lbs_in_water_sample_med", "lbs_in_water_sample_lo95",
                 "lbs_in_water_sample_hi95",
                 "lbs_in_water_med", "lbs_in_water_lo95", "lbs_in_water_hi95",
                 "pred_lbs_med", "pred_lbs_lo95", "pred_lbs_hi95",
                 "pred_lbs_reading_lo95", "pred_lbs_reading_hi95", "err_lbs"],
}


#: named presets for `view`
REPORT_VIEWS = {
    "compact": ["farm_season_id", "date", "event", "actual_lbs_ft",
                "pred_harvest_lbs_ft", "harvest_yield_lo95", "harvest_yield_hi95",
                "in95_yield", "in95_reading"],
    "yield":   ["keys", "context", "truth", "before", "after", "baseline", "verdict"],
    "pounds":  ["keys", "truth", "pounds", "verdict"],
    "all":     list(REPORT_GROUPS),
}


def season_report(events, fsid, prior_for, cache, sampler=None,
                  line_events=None, estimator="median", verbose=True):
    """
    One farm-season, one row per logged event, in chronological order.

    Its purpose is to replay the season: for every event, the forecast computed from everything
    logged strictly before it, and again after it is entered. The change between the two is how
    much that one observation taught the model.

    Outplant rows get forecasts too, from the posterior conditioned on whatever was logged strictly
    earlier (usually nothing). They are predictions at an outplant date, but nothing was fitted.

    Args:
        events (pd.DataFrame): events table with outplants, samples, and harvests.
        fsid (str): farm-season id to replay.
        prior_for (callable): Mapping of fsid to its model.Prior, e.g. from 
        `model.unpooled_prior_for` or `hierarchy.hierarchical_prior_for`.
        cache (model.PosteriorCache): supplies the posteriors. Every prefix of this farm-season
            is fetched, so the first call on a farm-season is slow and later ones are hits.
        sampler (dict | None): sampler settings overrides. Merged with the current sampler settings
            from cache.
        line_events (pd.DataFrame | None): outplant, line loss, and harvest events that affect
            current line length in water from `preprocess.load_line_events`. Without it
            `line_in_water_ft` is absent and the biomass-in-water view cannot be built.
        estimator (str): how draws become one number. Median by default.
        verbose (bool): print the cache hit/miss tally for this call.

    Returns:
        pd.DataFrame: one row per event in chronological order. With predicted harvest + 95 ranges
            before and after the event is logged.
    """
    obs = season_events(events, fsid)
    if not obs:
        raise ValueError(f"no observations for {fsid}")
    _before = cache.counters()
    g = events[events[IDCOL] == fsid]
    meta = g.iloc[0]
    prior = prior_for(fsid)
    c_param = prior["_meta"].get("c_param", cache.c_param)
    date_of = dict(zip(g[TCOL].astype(float),
                       pd.to_datetime(g["Log Date"]).dt.date))

    in_water = None
    if line_events is not None:
        from .preprocess import line_in_water
        in_water = lambda day: line_in_water(line_events, fsid, day)

    rows = []
    for d in sorted(g.loc[g.event == OUTPLANT_EVENT, TCOL].astype(float).unique()):
        n_before = sum(1 for od, *_ in obs if od < d)
        post_o = cache.get(fsid, obs, n_before, prior, sampler)

        (o_med, o_ylo, o_yhi), (_, o_rlo, o_rhi) = harvest_intervals(
            post_o, d, (fsid, d, n_before, "outplant"), c_param, cache.curve)

        os_med, os_lo, os_hi = sample_interval(post_o, d, cache.curve)

        rows.append({"farm_season_id": fsid, "date": date_of.get(d), "day": d,
                     "event": OUTPLANT_EVENT, "actual_lbs_ft": 0.0,
                     "n_samples_seen": sum(1 for e in obs[:n_before]
                                           if e[2] == SAMPLE_EVENT),
                     "n_harvests_seen": sum(1 for e in obs[:n_before]
                                            if e[2] == HARVEST_EVENT),
                     "pred_sample_lbs_ft": os_med,
                     "sample_yield_lo95": os_lo, "sample_yield_hi95": os_hi,
                     "pred_harvest_lbs_ft": o_med,
                     "harvest_yield_lo95": o_ylo, "harvest_yield_hi95": o_yhi,
                     "harvest_reading_lo95": o_rlo, "harvest_reading_hi95": o_rhi,
                     "line_in_water_ft": in_water(d + 1) if in_water else np.nan})


    for n, (day, actual, kind, line_ft, weight) in enumerate(obs):
        seen = obs[:n]
        post_b = cache.get(fsid, obs, n, prior, sampler)
        post_a = cache.get(fsid, obs, n + 1, prior, sampler)

        (b_med, b_ylo, b_yhi), (_, b_rlo, b_rhi) = harvest_intervals(
            post_b, day, (fsid, day, n, "before"), c_param, cache.curve)
        (a_med, a_ylo, a_yhi), (_, a_rlo, a_rhi) = harvest_intervals(
            post_a, day, (fsid, day, n + 1, "after"), c_param, cache.curve)

        s_med, s_lo, s_hi = sample_interval(post_b, day, cache.curve)
        sa_med, sa_lo, sa_hi = sample_interval(post_a, day, cache.curve)

        is_h = kind == HARVEST_EVENT

        rows.append({
            "farm_season_id": fsid, "date": date_of.get(day), "day": day,
            "event": kind,
            "n_samples_seen": sum(1 for e in seen if e[2] == SAMPLE_EVENT),
            "n_harvests_seen": sum(1 for e in seen if e[2] == HARVEST_EVENT),
            # days between the last observation the forecast saw and this day;
            # a large gap is why an interval is wide
            "gap_days": (day - seen[-1][0]) if seen else np.nan,
            "actual_lbs_ft": actual,
            "actual_lbs": weight if is_h else np.nan,
            "line_harvested_ft": line_ft if is_h else np.nan,
            "line_in_water_ft": in_water(day) if in_water else np.nan,
            "pred_sample_lbs_ft": s_med,
            "sample_yield_lo95": s_lo, "sample_yield_hi95": s_hi,
            "pred_sample_lbs_ft_after": sa_med,
            "sample_yield_lo95_after": sa_lo, "sample_yield_hi95_after": sa_hi,
            "pred_harvest_lbs_ft": b_med,
            "harvest_yield_lo95": b_ylo, "harvest_yield_hi95": b_yhi,
            "harvest_reading_lo95": b_rlo, "harvest_reading_hi95": b_rhi,
            "pred_harvest_lbs_ft_after": a_med,
            "harvest_yield_lo95_after": a_ylo, "harvest_yield_hi95_after": a_yhi,
            "harvest_reading_lo95_after": a_rlo, "harvest_reading_hi95_after": a_rhi,
            "naive_lbs_ft": naive_lbs_ft(seen, day),
            "in95_yield": (bool(b_ylo <= actual <= b_yhi) if is_h else None),
            "in95_reading": (bool(b_rlo <= actual <= b_rhi) if is_h else None),
        })

    t = pd.DataFrame(rows).sort_values(["day", "event"], kind="stable")
    t = t.reindex(columns=[c for grp in REPORT_GROUPS.values() for c in grp
                           if c in t.columns])

    if verbose:
        h = t[t["event"] == HARVEST_EVENT]
        print(f"{meta['Farm Name']} {meta['Season']} ({meta.get('State', '?')}) — "
              f"{len(h)} harvests, {int((t['event'] == SAMPLE_EVENT).sum())} samples"
              f"  [prior: {prior['_meta']['source']}]")
        if len(h):
            print(f"  harvest actuals inside the 95% yield range: "
                  f"{int(h['in95_yield'].sum())}/{len(h)}   "
                  f"reading range: {int(h['in95_reading'].sum())}/{len(h)}")
        cache.report_since(_before, f"season_report {fsid}")

    return t.reset_index(drop=True)



def scorable_farm_seasons(events, season=None, min_samples=0):
    """
    Farm-seasons a backtest can actually score (i.e. compare to GreenWave's method).

    A harvest needs both a logged weight and a line length to calculate its density.

    `min_samples=0` keeps the farm-seasons without samples but with a harvest.

    Args:
        events (pd.DataFrame): an events table with outplants, samples, and harvests.
        season (str | None): narrow to one season if desired.
        min_samples (int): require at least this many sample events within a farm-season. Default 0.

    Returns:
        list[str]: sorted farm-season ids based on their id.
    """
    d = events if season is None else events[events["Season"] == season]
    h = d[(d["event"] == HARVEST_EVENT) & d["weight_lbs"].notna()
          & d["line_ft"].notna()]

    fsids = set(h[IDCOL])
    if min_samples:
        n = d[d["event"] == SAMPLE_EVENT].groupby(IDCOL).size()
        fsids &= set(n[n >= min_samples].index)
    return sorted(fsids)


def harvest_predictions(events, fsids, prior_for, cache, estimator="median",
                        sampler=None, method=None, verbose=False):
    """
    One row per harvest event: what was predicted before it, and what happened.

    Each harvest is forecast from the observations strictly before its day, so a row never sees
    the value it predicts. Includes the predicted harvest in lbs/ft and the 95% ranges.

    Args:
        events (pd.DataFrame): a events table with outplants, samples, and harvests.
        fsids (list[str]): farm-seasons to get harvest predictions and compare
        prior_for (callable): Function that maps fsid to their `model.Prior`. Same one for unpooled
            model.
        cache (model.PosteriorCache): supplies one posterior per harvest.
        estimator (str): the point estimate to use on the posterior distribution. Defaults to
            `median`.
        sampler (dict | None): overrides merged over the cache's settings.
        method (str | None): label for the rows, so can compare results from different priors. 
            Defaults to the prior's own `source` string.
        verbose (bool): print the row count and the cache hit/miss tally.

    Returns:
        pd.DataFrame: one row per scored harvest. Includes predicted harvest + 95 ranges with other
            metadata about the prediction.
    """
    rows = []
    _before = cache.counters()

    for fsid in fsids:
        obs = season_events(events, fsid)
        g = events[events[IDCOL] == fsid]
        meta = g.iloc[0]
        prior = prior_for(fsid)
        c_param = prior["_meta"].get("c_param", cache.c_param)
        date_of = dict(zip(g[TCOL].astype(float),
                           pd.to_datetime(g["Log Date"]).dt.date))

        for n, (day, actual, kind, line_ft, weight) in enumerate(obs):
            if kind != HARVEST_EVENT or not np.isfinite(line_ft):
                continue
            seen = obs[:n]
            post = cache.get(fsid, obs, n, prior, sampler)

            (med, ylo, yhi), (_, rlo, rhi) = harvest_intervals(
                post, day, (fsid, day, n, "before"), c_param, cache.curve)

            if estimator == "joint_mode":
                med = joint_mode_prediction(post, day, c_param, cache.curve)
            elif estimator == "mean":
                med = point_estimate(post, day, "mean", c_param, cache.curve)[1]

            nv = naive_lbs_ft(seen, day)

            rows.append({
                "method": method or prior["_meta"]["source"],
                "farm_season_id": fsid, "farm": meta["Farm Name"],
                "season": meta["Season"], "date": date_of.get(day), "day": day,
                "estimator": estimator,
                "n_samples_seen": sum(1 for e in seen if e[2] == SAMPLE_EVENT),
                "n_harvests_seen": sum(1 for e in seen if e[2] == HARVEST_EVENT),
                "gap_days": (day - seen[-1][0]) if seen else np.nan,
                "actual_lbs_ft": actual, "actual_lbs": weight,
                "line_harvested_ft": line_ft,
                "pred_lbs_ft": med,
                "yield_lo95": ylo, "yield_hi95": yhi,
                "reading_lo95": rlo, "reading_hi95": rhi,
                "in95_yield": bool(ylo <= actual <= yhi),
                "in95_reading": bool(rlo <= actual <= rhi),
                "naive_lbs_ft": nv,
                "pred_lbs": med * line_ft,
                "naive_lbs": nv * line_ft,
                "err_lbs_ft": med - actual,
                "naive_err_lbs_ft": nv - actual,
                "err_lbs": med * line_ft - weight,
                "naive_err_lbs": nv * line_ft - weight,
            })

    harvest_comparison_table = pd.DataFrame(rows)
    if verbose and len(harvest_comparison_table):
        print(f"{len(harvest_comparison_table)} harvest events over "
              f"{harvest_comparison_table['farm_season_id'].nunique()} "
              f"farm-seasons [estimator={estimator}]")
        cache.report_since(_before, "harvest_predictions")
    return harvest_comparison_table


def by_farm_season(preds, with_naive=True):
    """
    Roll harvest predictions up to season totals.

    Reports the total predicted harvest and `n_in95_yield` and `n_in95_reading` which counts the
    number of harvests lbs/ft within the 95% interval range of the posterior distribution draws.

    Args:
        preds (pd.DataFrame): `harvest_predictions` output. Grouped by `method`
            as well when that column is present.
        with_naive (bool): Whether to compare the harvest total predictions against GreenWave's
            appraoch. Include the baseline totals, errors and `winner`. Defaults to True.

    Returns:
        pd.DataFrame: one row per farm-season (per method).
    """
    n_all = preds.groupby("farm_season_id")["actual_lbs"].size()
    keys = ["farm_season_id", "farm", "season"]
    if "method" in preds.columns:
        keys = ["method"] + keys
    agg = {"n_harvests": ("actual_lbs", "size"),
           "actual_lbs": ("actual_lbs", "sum"),
           "pred_lbs": ("pred_lbs", "sum")}

    if "in95_yield" in preds.columns:
        agg["n_in95_yield"] = ("in95_yield", "sum")
    if "in95_reading" in preds.columns:
        agg["n_in95_reading"] = ("in95_reading", "sum")
    if with_naive:
        agg["naive_lbs"] = ("naive_lbs", "sum")
        agg["n_harvests_naive"] = ("naive_lbs", "count")

    t = preds.groupby(keys, dropna=False).agg(**agg).reset_index()
    t["n_harvests_in_season"] = t["farm_season_id"].map(n_all)
    for c in ("n_in95_yield", "n_in95_reading"):
        if c in t.columns:
            t[c] = t[c].astype(int)
    t["err_lbs"] = t["pred_lbs"] - t["actual_lbs"]
    t["abs_err_lbs"] = t["err_lbs"].abs()
    t["pct_err"] = 100 * t["err_lbs"] / t["actual_lbs"].where(t["actual_lbs"] > 0)
    if with_naive:
        t["naive_err_lbs"] = t["naive_lbs"] - t["actual_lbs"]
        t["naive_abs_err_lbs"] = t["naive_err_lbs"].abs()
        t["naive_pct_err"] = 100 * t["naive_err_lbs"] / t["actual_lbs"].where(t["actual_lbs"] > 0)
        t["winner"] = np.where(t["abs_err_lbs"] < t["naive_abs_err_lbs"], "model",
                       np.where(t["naive_abs_err_lbs"] < t["abs_err_lbs"], "naive", "tie"))
    return t


def evaluate(preds):
    """
    Headline metrics per method: yield error, total biomass error, coverage, and farm-seasons won.

    Every scorable harvest is included. Both a mean and a median are given for each error which can
    disagree.

    Args:
        preds (pd.DataFrame): `harvest_predictions` output. Can be several methods concatenated.

    Returns:
        pd.DataFrame: one row per method, indexed by `method`.
    """
    groups = ([(m, g) for m, g in preds.groupby("method")]
              if "method" in preds.columns else [("model", preds)])

    rows = []
    for name, g in groups:
        tot = by_farm_season(g)
        rows.append({
            "method": name,
            "harvests": len(g), "farm_seasons": g["farm_season_id"].nunique(),
            "yield_mae": g["err_lbs_ft"].abs().mean(),
            "yield_medae": g["err_lbs_ft"].abs().median(),
            "naive_yield_mae": g["naive_err_lbs_ft"].abs().mean(),
            "naive_yield_medae": g["naive_err_lbs_ft"].abs().median(),
            "total_mae_lbs": tot["abs_err_lbs"].mean(),
            "total_medae_lbs": tot["abs_err_lbs"].median(),
            "naive_total_mae_lbs": tot["naive_abs_err_lbs"].mean(),
            "naive_total_medae_lbs": tot["naive_abs_err_lbs"].median(),
            "total_pct_err_pooled": 100 * (tot["pred_lbs"].sum() - tot["actual_lbs"].sum())
                                    / tot["actual_lbs"].sum(),
            "naive_total_pct_err_pooled": 100 * (tot["naive_lbs"].sum() - tot["actual_lbs"].sum())
                                          / tot["actual_lbs"].sum(),
            "in95_yield_pct": 100 * g["in95_yield"].mean(),
            "in95_reading_pct": 100 * g["in95_reading"].mean(),
            "farm_seasons_won": int((tot["winner"] == "model").sum()),
            "farm_seasons_lost": int((tot["winner"] == "naive").sum()),
        })
    return pd.DataFrame(rows).set_index("method")


def harvest_coverage(df, fsids, prior_for, cache, mcmc=None, quiet=False,
                     label="model"):
    """
    Answers how often the actual harvest landed inside the 95% range predicted before it.

    Two ranges:
    - `in95_yield` is the range for the harvest yield, `c * f(day)`, the
    farm's average that day; it is narrower. 
    - `in95_reading` adds `sigma_h` to include the scatter of the true harvest around the curve.

    Args:
        df (pd.DataFrame): events table with outplants, samples, and harvests.
        fsids (list[str]): farm-seasons to score.
        prior_for (callable): Mapping of fsid to its `model.Prior`.
        cache (model.PosteriorCache): supplies the posteriors.
        mcmc (dict | None): sampler settings overrides. Merged with the cache's current settings.
        quiet (bool): skip the printed summary. Defaults to False.
        label (str): name used in that printout. Defaults to "model".

    Returns:
        tuple[pd.DataFrame, dict]: one row per harvest with the verdict if within the ranges and the
            ranges themselves.
    """
    rows = []
    _before = cache.counters()

    for fsid in fsids:
        obs = season_events(df, fsid)
        prior = prior_for(fsid)
        c_param = prior["_meta"].get("c_param", cache.c_param)
        meta = df[df[IDCOL] == fsid].iloc[0]

        for n, (day, actual, kind, line_ft, weight) in enumerate(obs):
            if kind != HARVEST_EVENT:
                continue
            post = cache.get(fsid, obs, n, prior, mcmc)
            (e_med, e_lo, e_hi), (_, p_lo, p_hi) = harvest_intervals(
                post, day, (fsid, day, n, "before"), c_param, cache.curve)
            seen = obs[:n]

            rows.append(dict(
                farm_season_id=fsid, farm=meta["Farm Name"],
                season=meta["Season"], day=day, actual_lbs_ft=actual,
                line_ft=line_ft, actual_lbs=weight,
                n_seen_samples=sum(1 for e in seen if e[2] == SAMPLE_EVENT),
                n_seen_harvests=sum(1 for e in seen if e[2] == HARVEST_EVENT),
                pred_med=e_med, yield_lo95=e_lo, yield_hi95=e_hi,
                reading_lo95=p_lo, reading_hi95=p_hi,
                in95_yield=bool(e_lo <= actual <= e_hi),
                in95_reading=bool(p_lo <= actual <= p_hi),
                miss=("" if p_lo <= actual <= p_hi
                      else ("above" if actual > p_hi else "below")),
                rel_width_yield=((e_hi - e_lo) / e_med if e_med > 0 else np.nan),
                rel_width_reading=((p_hi - p_lo) / e_med if e_med > 0 else np.nan)))

    t = pd.DataFrame(rows)
    if not len(t):
        return t, {}

    summary = {
        "harvests": len(t),
        "farm_seasons": t["farm_season_id"].nunique(),
        "in 95% (reading interval)": 100 * t["in95_reading"].mean(),
        "in 95% (yield interval)": 100 * t["in95_yield"].mean(),
        "median relative width (reading)": t["rel_width_reading"].median(),
        "median relative width (yield)": t["rel_width_yield"].median(),
        "misses above": int((t["miss"] == "above").sum()),
        "misses below": int((t["miss"] == "below").sum()),
    }

    if not quiet:
        print(f"[{label}] {summary['harvests']} harvests over "
              f"{summary['farm_seasons']} farm-seasons")
        cache.report_since(_before, "harvest_coverage")
        print(f"  inside 95% reading interval: "
              f"{summary['in 95% (reading interval)']:.1f}%  "
              f"(median width {summary['median relative width (reading)']:.1f}x "
              f"the prediction)")
        print(f"  inside 95% yield interval:   "
              f"{summary['in 95% (yield interval)']:.1f}%  "
              f"(median width {summary['median relative width (yield)']:.1f}x)")
        print(f"  misses: {summary['misses above']} above, "
              f"{summary['misses below']} below")

    return t, summary


def describe_farm_seasons(events, season=None, scorable_only=True):
    """
    What is in each farm-season including n_samples, n_harvests.

    Sorted by `balance`, the weaker of the sample and harvest counts, to measure how "illustrative"
    a farm-season is. If a farm-season has lots of samples and harvests, the posterior will have
    more data to see and should have a narrower distribution as the observations accumulate
    for both samples and harvests.

    Args:
        events (pd.DataFrame): events table with outplants, samples, and harvests.
        season (str | None): narrow to one season if desired.
        scorable_only (bool): restrict to `scorable_farm_seasons`. Defaults to True.

    Returns:
        pd.DataFrame: one row per farm-season indexed by id, most illustrative first.
    """
    d = events if season is None else events[events["Season"] == season]
    keep = set(scorable_farm_seasons(d)) if scorable_only else set(d[IDCOL])
    d = d[d[IDCOL].isin(keep)]
    obs = d[d["event"].isin([SAMPLE_EVENT, HARVEST_EVENT])]

    g = obs.groupby(IDCOL)
    t = pd.DataFrame({
        "farm": g["Farm Name"].first(),
        "season": g["Season"].first(),
        "state": g["State"].first() if "State" in d.columns else "",
        "n_samples": obs[obs.event == SAMPLE_EVENT].groupby(IDCOL).size(),
        "n_harvests": obs[obs.event == HARVEST_EVENT].groupby(IDCOL).size(),
        "actual_lbs": d[d.event == HARVEST_EVENT].groupby(IDCOL)["weight_lbs"].sum(),
        "first_day": g[TCOL].min(),
        "last_day": g[TCOL].max(),
    })

    t[["n_samples", "n_harvests"]] = t[["n_samples", "n_harvests"]].fillna(0).astype(int)
    t["span_days"] = t["last_day"] - t["first_day"]

    t["balance"] = t[["n_samples", "n_harvests"]].min(axis=1)
    return t.sort_values(["balance", "actual_lbs"], ascending=False)
