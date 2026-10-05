"""
Generate farm_season_data.json for the React app.

This script replaces the chart-rendering parts of build_interactive.py.
It extracts the same data and tags, but writes them as JSON instead of HTML.
The React app reads this JSON file and renders the charts at runtime.

Run this whenever you have new data:
    python scripts/export_data.py

Output: public/farm_season_data.json
"""

import json
import re
from pathlib import Path

import pandas as pd

# --- Config ---
DATA_FILE = Path("../GreenWave_anon_kcf_export_20260416.xlsx")
OUTPUT_FILE = Path("../public/farm_season_data.json")

# --- Load and parse (same as build_interactive.py) ----------------------
print(f"Reading {DATA_FILE}...")
ALL = pd.read_excel(DATA_FILE, sheet_name=None)
logs = ALL["Logs"].loc[:, ~ALL["Logs"].columns.str.startswith("Unnamed")]
logs["Log Date"] = pd.to_datetime(logs["Log Date"])


def extract_line_numbers(note):
    """Pull line numbers out of free-text Notes (e.g. 'line 3 and 5')."""
    found = []
    if isinstance(note, str):
        for match in re.finditer(r"\blines?\b[\s:=>/]*([\d,\s&]+(?:and[\s\d,&]+)*)", note.lower()):
            for c in re.findall(r"\d+", match.group(1)):
                num = int(c)
                if num not in found:
                    found.append(num)
        found.sort()
    return found


logs["line_numbers"] = logs["Notes"].apply(extract_line_numbers)
logs["n_lines"] = logs["line_numbers"].apply(len)


def line_key(row):
    """Label a log row by its line: one number, several ("multi: ..."), or "unspecified"."""
    if row["n_lines"] == 0:
        return "unspecified"
    if row["n_lines"] == 1:
        return str(row["line_numbers"][0])
    return "multi: " + ",".join(map(str, row["line_numbers"]))


sh = logs[logs["Log Type"].isin(["sample", "harvest"])].copy()
sh["line_key"] = sh.apply(line_key, axis=1)
sh["species_key"] = sh["Species"].fillna("unspecified")


# --- Tag computation (mirrors build_interactive.py) ---------------------
def panel_tags(n_samples, n_harvests, line_str, samples_df, harvests_df):
    """Tags for a single (species, line) panel."""
    tags = []
    if line_str == "unspecified":
        tags.append("no-line")
    elif line_str.startswith("multi"):
        tags.append("multi-line")
    else:
        tags.append("line-specified")
    tags.append(f"{n_samples}-samples" if n_samples != 1 else "1-sample")
    tags.append(f"{n_harvests}-harvests" if n_harvests != 1 else "1-harvest")
    if n_samples and n_harvests:
        if (samples_df["Log Date"] < harvests_df["Log Date"].min()).sum() == 0:
            tags.append("no-pre-harvest-sample")
    return tags


def rollup_tags(g, panel_tag_lists):
    """Tags applied to a whole farm-season section, used for filtering and display.

    The bulk of the tags come from a UNION of all per-plot tags inside this
    farm-season (`panel_tag_lists`). On top of that we add a couple of tags
    that only make sense at the farm-season level (multi-species, has-both).
    """
    tags = set()
    # Union of all per-plot tags
    for panel_tags in panel_tag_lists:
        tags.update(panel_tags)
    # Species variety (only meaningful at farm-season level)
    species = g["species_key"][g["species_key"] != "unspecified"].unique()
    if len(species) >= 2:  # noqa: PLR2004 -- "multi" means more than one
        tags.add("multi-species")
    # Analysis-ready: this farm-season has at least 1 sample AND at least 1 harvest
    # (somewhere among its plots, not necessarily in the same plot)
    samples = g[g["Log Type"] == "sample"]
    harvests = g[g["Log Type"] == "harvest"]
    if len(samples) > 0 and len(harvests) > 0:
        tags.add("has-both")
    return sorted(tags)


# --- Build farm-season records ------------------------------------------
print("Building farm-season records...")


def date_str(d):
    """Convert pandas Timestamp to ISO date string (or None if NaN)."""
    if pd.isna(d):
        return None
    return pd.Timestamp(d).strftime("%Y-%m-%d")


farm_seasons = []
for (farm, season), g in sh.groupby(["Anon Farm", "Season"]):
    g = g.sort_values("Log Date").reset_index(drop=True)

    # Identify the panels (unique species x line combinations in this farm-season)
    facets = []
    for species in g["species_key"].unique():
        for line in g[g["species_key"] == species]["line_key"].unique():
            facets.append((species, line))

    panels = []
    for species, line in facets:
        sub = g[(g["species_key"] == species) & (g["line_key"] == line)]
        samples = sub[sub["Log Type"] == "sample"].sort_values("Log Date")
        harvests = sub[sub["Log Type"] == "harvest"].sort_values("Log Date")

        # Convert each sample to a small dict
        sample_records = []
        for _, r in samples.iterrows():
            sample_records.append(
                {
                    "date": date_str(r["Log Date"]),
                    "weight": None if pd.isna(r["Weight"]) else float(r["Weight"]),
                    "notes": str(r["Notes"])[:140] if pd.notna(r["Notes"]) else "",
                }
            )

        # Convert each harvest to a small dict (compute yield if possible)
        harvest_records = []
        for _, r in harvests.iterrows():
            line_length = None if pd.isna(r["Line Length"]) else float(r["Line Length"])
            weight = None if pd.isna(r["Weight"]) else float(r["Weight"])
            yield_val = None
            if weight is not None and line_length is not None and line_length > 0:
                yield_val = weight / line_length
            harvest_records.append(
                {
                    "date": date_str(r["Log Date"]),
                    "weight": weight,
                    "line_length": line_length,
                    "yield_lb_per_ft": yield_val,
                    "notes": str(r["Notes"])[:140] if pd.notna(r["Notes"]) else "",
                }
            )

        panels.append(
            {
                "species": species,
                "line": line,
                "n_samples": len(samples),
                "n_harvests": len(harvests),
                "tags": panel_tags(len(samples), len(harvests), line, samples, harvests),
                "samples": sample_records,
                "harvests": harvest_records,
            }
        )

    farm_seasons.append(
        {
            "farm": farm,
            "season": season,
            "id": f"{farm}__{season}".replace(" ", "_").replace("/", "-"),
            "tags": rollup_tags(g, [p["tags"] for p in panels]),
            "n_panels": len(panels),
            "panels": panels,
        }
    )

# Sort: farm name, then season
farm_seasons.sort(key=lambda x: (x["farm"], x["season"]))

# --- Build a global summary ---------------------------------------------
all_tags = set()
for fs in farm_seasons:
    all_tags.update(fs["tags"])
    for p in fs["panels"]:
        all_tags.update(p["tags"])

output = {
    "metadata": {
        "n_farm_seasons": len(farm_seasons),
        "n_farms": len({fs["farm"] for fs in farm_seasons}),
        "all_tags": sorted(all_tags),
        "generated_from": str(DATA_FILE.name),
    },
    "farm_seasons": farm_seasons,
}

# --- Write JSON ---------------------------------------------------------
OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
with OUTPUT_FILE.open("w") as f:
    json.dump(output, f, separators=(",", ":"))  # minified

size_kb = OUTPUT_FILE.stat().st_size / 1024
print(f"Wrote {OUTPUT_FILE} ({size_kb:.0f} KB)")
print(f"  {len(farm_seasons)} farm-seasons, {len(all_tags)} distinct tags")
