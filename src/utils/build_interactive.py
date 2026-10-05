"""
Interactive farm-season HTML using Plotly.

Fixes from v1:
  - Legend has fixed entries (Samples / Daily-average / Harvest) regardless of harvest count
  - Hover on any point shows exact date + value
  - Plotly CDN included once, all figures embed as small divs (file is ~1-2 MB instead of 14 MB)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- Load + parse (mirrors EDA notebook) --------------------------------
ALL = pd.read_excel("/home/claude/data.xlsx", sheet_name=None)
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


# -- Per-farm-season figure ----------------------------------------------
COLOR_SAMPLE = "#1D9E75"
COLOR_DAILY = "#0F6E50"
COLOR_HARVEST = "#D85A30"


def panel_tags(n_samples, n_harvests, line_str, samples_df, harvests_df):
    """Tags applied to a single (species, line) panel."""
    tags = []
    # Line specification status
    if line_str == "unspecified":
        tags.append("no-line")
    elif line_str.startswith("multi"):
        tags.append("multi-line")
    else:
        tags.append("line-specified")
    # Exact sample count (no buckets - show the actual number)
    tags.append(f"{n_samples}-samples" if n_samples != 1 else "1-sample")
    # Exact harvest count
    tags.append(f"{n_harvests}-harvests" if n_harvests != 1 else "1-harvest")
    # Sample timing
    if n_samples and n_harvests:
        if (samples_df["Log Date"] < harvests_df["Log Date"].min()).sum() == 0:
            tags.append("no-pre-harvest-sample")
    return tags


def make_farm_season_figure(farm, season):
    """One farm-season's figure, a panel per (species, line), and the tags of its panels."""
    g = sh[(sh["Anon Farm"] == farm) & (sh["Season"] == season)]
    if g.empty:
        return None, []

    facets = sorted(g.groupby(["species_key", "line_key"]).groups.keys())
    n = len(facets)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols

    subplot_titles = []
    facet_tags_all = set()  # union of tags across this farm-season (for header)
    for sp, ln in facets:
        sub = g[(g["species_key"] == sp) & (g["line_key"] == ln)]
        s_ = sub[sub["Log Type"] == "sample"]
        h_ = sub[sub["Log Type"] == "harvest"]
        tags = panel_tags(len(s_), len(h_), ln, s_, h_)
        facet_tags_all.update(tags)
        line_label = "no line specified" if ln == "unspecified" else f"line {ln}"
        subplot_titles.append(
            f"<b>{sp} / {line_label}</b>  (s={len(s_)}, h={len(h_)})<br>"
            f"<span style='font-size:10px;color:#888'>{' · '.join(tags)}</span>"
        )

    # Vertical spacing budget per row boundary (verified visually, not by math):
    # the rotated date tick labels are taller than they look (60-70px), and the
    # row above also has an "Date" axis title (~20px below that). Add the row
    # below's 2-line subplot title (~35px) plus breathing room (~25px).
    GAP_PX = 150  # pixels of vertical gap per row boundary
    PLOT_PX_PER_ROW = 320  # plot area per row
    plot_height_px = rows * PLOT_PX_PER_ROW + max(rows - 1, 0) * GAP_PX
    v_space = (GAP_PX / plot_height_px) if rows > 1 else 0.1
    # plotly hard-caps vertical_spacing at 1/(rows-1)
    v_space = min(v_space, 1 / max(rows - 1, 1) - 0.01) if rows > 1 else v_space
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=v_space,
        horizontal_spacing=0.08,
    )

    # We want exactly one legend entry per kind, regardless of how many facets
    sample_legend_shown = False
    daily_legend_shown = False
    harvest_legend_shown = False

    for i, (species, line) in enumerate(facets):
        r = i // cols + 1
        c = i % cols + 1
        sub = g[(g["species_key"] == species) & (g["line_key"] == line)]
        samples = sub[sub["Log Type"] == "sample"].sort_values("Log Date")
        harvests = sub[sub["Log Type"] == "harvest"].sort_values("Log Date")

        # Raw sample dots
        if len(samples):
            # Truncate notes for hover readability
            notes_text = samples["Notes"].fillna("").astype(str).str.slice(0, 140)
            fig.add_trace(
                go.Scatter(
                    x=samples["Log Date"],
                    y=samples["Weight"],
                    mode="markers",
                    name="Samples",
                    legendgroup="samples",
                    showlegend=not sample_legend_shown,
                    marker={
                        "color": COLOR_SAMPLE,
                        "size": 8,
                        "opacity": 0.55,
                        "line": {"color": "white", "width": 0.6},
                    },
                    customdata=notes_text.to_numpy().reshape(-1, 1),
                    hovertemplate=(
                        "<b>Sample</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Weight: %{y:.2f} lb/ft<br>"
                        "Notes: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )
            sample_legend_shown = True

        # Daily-averaged line (only if ≥2 unique days)
        if len(samples) >= 2:  # noqa: PLR2004 -- a line needs two points
            tmp = samples.copy()
            tmp["date_only"] = tmp["Log Date"].dt.date
            daily = (
                tmp.groupby("date_only")
                .agg(mean_w=("Weight", "mean"), n=("Weight", "count"))
                .reset_index()
            )
            daily["date"] = pd.to_datetime(daily["date_only"])
            if len(daily) >= 2:  # noqa: PLR2004 -- a line needs two points
                fig.add_trace(
                    go.Scatter(
                        x=daily["date"],
                        y=daily["mean_w"],
                        mode="lines+markers",
                        name="Daily-average sample",
                        legendgroup="daily",
                        showlegend=not daily_legend_shown,
                        line={"color": COLOR_DAILY, "width": 2},
                        marker={"color": COLOR_DAILY, "size": 5},
                        customdata=daily[["n"]].values,
                        hovertemplate=(
                            "<b>Daily average</b><br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Mean weight: %{y:.2f} lb/ft<br>"
                            "n samples that day: %{customdata[0]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
                daily_legend_shown = True

        # Harvests — single trace per facet, no per-point labels
        if len(harvests):
            h_dates = []
            h_yields = []
            h_weights = []
            h_lengths = []
            h_notes = []
            no_yield_dates = []
            for _, h in harvests.iterrows():
                if pd.notna(h["Line Length"]) and h["Line Length"] > 0 and pd.notna(h["Weight"]):
                    h_dates.append(h["Log Date"])
                    h_yields.append(h["Weight"] / h["Line Length"])
                    h_weights.append(h["Weight"])
                    h_lengths.append(h["Line Length"])
                    note = h.get("Notes", "")
                    h_notes.append(str(note)[:140] if pd.notna(note) else "")
                else:
                    no_yield_dates.append(h["Log Date"])

            if h_dates:
                customdata = np.column_stack([h_weights, h_lengths, h_notes])
                fig.add_trace(
                    go.Scatter(
                        x=h_dates,
                        y=h_yields,
                        mode="markers",
                        name="Harvest (yield)",
                        legendgroup="harvest",
                        showlegend=not harvest_legend_shown,
                        marker={
                            "color": COLOR_HARVEST,
                            "size": 13,
                            "symbol": "square",
                            "line": {"color": "white", "width": 1},
                        },
                        customdata=customdata,
                        hovertemplate=(
                            "<b>Harvest</b><br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Yield: %{y:.2f} lb/ft<br>"
                            "Total weight: %{customdata[0]:.0f} lb<br>"
                            "Line length: %{customdata[1]:.0f} ft<br>"
                            "Notes: %{customdata[2]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
                harvest_legend_shown = True

            # Harvests with no line length → vertical dashed lines (no legend entry)
            for d in no_yield_dates:
                fig.add_vline(
                    x=d,
                    line={"color": COLOR_HARVEST, "dash": "dash", "width": 1},
                    row=r,
                    col=c,
                )

        fig.update_xaxes(title_text="Date", row=r, col=c, tickangle=-30)
        fig.update_yaxes(title_text="Weight (lb/ft)", row=r, col=c)

    # Total figure height: the plot area (computed above based on row count + gap budget)
    # plus the top margin (room for figure title + legend strip).
    top_margin = 110  # 24 figure title + ~28 legend strip + spacing
    total_height = plot_height_px + top_margin
    fig.update_layout(
        title={
            "text": f"<b>{farm} — {season}</b>",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 13},
        },
        height=total_height,
        margin={"l": 50, "r": 20, "t": top_margin, "b": 50},
        legend={
            "orientation": "h",
            "yref": "container",
            "yanchor": "top",
            "y": 1 - 32 / total_height,  # 32px below top
            "xanchor": "right",
            "x": 0.99,
            "bgcolor": "rgba(255,255,255,0.85)",
            "font": {"size": 10},
            "tracegroupgap": 4,
        },
        hovermode="closest",
        template="plotly_white",
        font={"size": 11},
    )
    # Make subplot title fonts smaller
    for ann in fig["layout"]["annotations"]:
        ann["font"] = {"size": 11}

    return fig, sorted(facet_tags_all)


# -- Build full page -----------------------------------------------------
print("Generating figures…")
fs_pairs = (
    sh.groupby(["Anon Farm", "Season"])
    .agg(
        n_samples=("Log Type", lambda x: (x == "sample").sum()),
        n_harvests=("Log Type", lambda x: (x == "harvest").sum()),
        n_species=("species_key", "nunique"),
        n_lines=("line_key", "nunique"),
    )
    .reset_index()
)
# Sort farms numerically (Farm_1, Farm_2, ... Farm_10, not Farm_1, Farm_10, Farm_11)
fs_pairs["_farm_num"] = fs_pairs["Anon Farm"].str.extract(r"(\d+)").astype(int)
fs_pairs = fs_pairs.sort_values(["_farm_num", "Season"]).drop(columns=["_farm_num"])

user_df = ALL["User"].set_index("Anon Farm")


# Pre-compute per-farm-season stats needed for the rollup tag set
def farm_season_rollup_tags(farm, season):
    """Tags applied at the farm-season level (rolled up from facets)."""
    g = sh[(sh["Anon Farm"] == farm) & (sh["Season"] == season)]
    samples = g[g["Log Type"] == "sample"]
    harvests = g[g["Log Type"] == "harvest"]
    tags = []
    # Sample volume
    # Sample volume buckets (used for filtering -- exact count would create
    # too many tag categories to be useful as filters)
    if len(samples) == 0:
        tags.append("0-samples")
    elif len(samples) == 1:
        tags.append("1-sample")
    elif len(samples) < 5:  # noqa: PLR2004 -- matches the "5+samples" tag
        tags.append(f"{len(samples)}-samples")
    else:
        tags.append("5+samples")
    # Harvest volume
    if len(harvests) == 0:
        tags.append("0-harvests")
    elif len(harvests) == 1:
        tags.append("1-harvest")
    else:
        tags.append("2+harvests")
    # Line specification — at least one event has a real line number?
    has_line = (g["line_key"] != "unspecified") & (~g["line_key"].str.startswith("multi"))
    if has_line.any():
        tags.append("any-line-specified")
    else:
        tags.append("no-line-info")
    # Species variety
    species = g["species_key"][g["species_key"] != "unspecified"].unique()
    if len(species) >= 2:  # noqa: PLR2004 -- "multi" means more than one
        tags.append("multi-species")
    # Analysis-ready: has at least 1 sample AND at least 1 harvest
    if len(samples) > 0 and len(harvests) > 0:
        tags.append("has-both")  # renamed from samples+harvest for clarity
        # Sample-before-harvest?
        if (samples["Log Date"] < harvests["Log Date"].min()).sum() == 0:
            tags.append("no-pre-harvest-sample")
    return tags


nav_items = []
sections = []

for _, r in fs_pairs.iterrows():
    farm, season = r["Anon Farm"], r["Season"]
    anchor = f"{farm}_{season}".replace(" ", "_").replace("/", "-")

    rollup = farm_season_rollup_tags(farm, season)
    nav_items.append(
        f'<li><a href="#{anchor}">{farm} — {season}</a> '
        f'<span class="meta">({int(r["n_samples"])}s / {int(r["n_harvests"])}h)</span></li>'
    )

    meta_html = ""
    if farm in user_df.index:
        meta = user_df.loc[farm]
        meta_html = (
            f'<div class="farm-meta">'
            f"<b>Location:</b> {meta.get('City', '?')}, {meta.get('State', '?')}, {meta.get('Country', '?')} &nbsp; "
            f"<b>Acres:</b> {meta.get('Farm Acres', '?')}"
            f"</div>"
        )

    result = make_farm_season_figure(farm, season)
    if result is None or result[0] is None:
        plot_html = "<i>No plottable data</i>"
        facet_tags = []
    else:
        fig, facet_tags = result
        plot_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,  # Plotly loaded once in <head>
            config={"displaylogo": False, "responsive": True},
        )

    # Show: real line numbers seen (excluding "unspecified" and "multi: …"), or "none"
    g_fs = sh[(sh["Anon Farm"] == farm) & (sh["Season"] == season)]
    specified_lines = sorted(
        {
            ln
            for ln in g_fs["line_key"].unique()
            if ln != "unspecified" and not ln.startswith("multi")
        },
        key=lambda x: int(x),
    )
    if specified_lines:
        lines_str = ", ".join(specified_lines)
    else:
        lines_str = "<i>none specified</i>"

    species_list = sorted({s for s in g_fs["species_key"].unique() if s != "unspecified"})
    species_str = ", ".join(species_list) if species_list else "<i>none specified</i>"

    chip_html = "".join(f'<span class="chip chip-{t}">{t}</span>' for t in rollup)

    sections.append(
        f"""
        <section id="{anchor}" class="farm-season" data-tags="{' '.join(rollup)}">
          <h2>{farm} &mdash; {season}</h2>
          {meta_html}
          <div class="stats">
            Samples: {int(r["n_samples"])} &middot;
            Harvests: {int(r["n_harvests"])} &middot;
            Species: {species_str} &middot;
            Lines: {lines_str}
          </div>
          <div class="tags">{chip_html}</div>
          {plot_html}
        </section>
        """
    )

css = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
       margin: 0 auto; padding: 24px; color: #222; background: #fafafa; }
header { border-bottom: 2px solid #1D9E75; padding-bottom: 12px; margin-bottom: 16px; background: white; padding: 16px; border-radius: 6px;}
h1 { margin: 0; }
.summary { background: #f5f5f5; padding: 12px; border-radius: 6px; margin-top: 12px; font-size: 13px; }
.toc { background: white; padding: 12px 18px; border-radius: 6px;
       max-height: 320px; overflow-y: auto; margin-bottom: 24px;
       column-count: 2; column-gap: 24px; border: 1px solid #eee; }
.toc ul { margin: 0; padding-left: 18px; }
.toc li { font-size: 13px; line-height: 1.5; break-inside: avoid; }
.toc a { text-decoration: none; color: #1D9E75; }
.toc a:hover { text-decoration: underline; }
.meta { color: #888; font-size: 11px; }
.farm-season { border: 1px solid #e0e0e0; border-radius: 8px;
               padding: 16px; margin-bottom: 24px; background: white; }
.farm-season h2 { margin-top: 0; color: #1D9E75; }
.farm-meta { color: #555; font-size: 13px; margin-bottom: 6px; }
.stats { color: #444; font-size: 13px; margin-bottom: 10px; }
.filter-bar { position: sticky; top: 0; background: white;
              padding: 10px; z-index: 100; border-bottom: 1px solid #eee;
              margin-bottom: 12px; border-radius: 6px; }
.filter-bar input { padding: 6px 10px; width: 320px; font-size: 14px;
                    border: 1px solid #ccc; border-radius: 4px; }
.tags { margin: 8px 0 14px; }
.chip { display: inline-block; padding: 2px 9px; margin: 2px 4px 2px 0;
        border-radius: 11px; font-size: 11px; background: #eef3f0;
        color: #2a5a47; border: 1px solid #d8e5dc; }
/* Quality-tier color hints */
.chip-well-sampled, .chip-multi-harvest, .chip-line-specified,
.chip-any-line-specified, .chip-samples\\+harvest {
  background: #def5e8; color: #1b6a3f; border-color: #b8e6cb; }
.chip-no-line, .chip-no-line-info, .chip-1-sample, .chip-no-samples,
.chip-no-harvest, .chip-no-pre-harvest-sample {
  background: #fce9d8; color: #8a4a1e; border-color: #f0cba5; }
.chip-few-samples, .chip-single-harvest, .chip-multi-line, .chip-multi-species {
  background: #e5e8f5; color: #3b3d80; border-color: #c8ccea; }

.tag-filters { background: white; padding: 10px 14px; border-radius: 6px;
               margin-bottom: 12px; border: 1px solid #eee; font-size: 13px; }
.tag-filters strong { margin-right: 8px; color: #444; }
.tag-filters button { font-size: 12px; padding: 3px 10px; margin: 2px 3px;
                      border-radius: 11px; border: 1px solid #ccc;
                      background: white; cursor: pointer; }
.tag-filters button.active { background: #1D9E75; color: white; border-color: #1D9E75; }
.tag-filters .clear { color: #888; }
.filter-bar .count { margin-left: 12px; color: #888; font-size: 12px; }
"""

js = """
const input = document.getElementById('search');
const sections = document.querySelectorAll('section.farm-season');
const tocItems = document.querySelectorAll('.toc li');
const counter = document.getElementById('match-count');
const tagButtons = document.querySelectorAll('.tag-filters button[data-tag]');
const clearBtn = document.getElementById('clear-tags');
const total = sections.length;
const activeTags = new Set();

function applyFilter() {
    const q = input.value.toLowerCase();
    let shown = 0;
    sections.forEach((s, i) => {
        const sectionTags = (s.dataset.tags || '').split(' ');
        const textMatch = s.textContent.toLowerCase().includes(q);
        // Section must have EVERY active tag (AND-logic)
        const tagMatch = [...activeTags].every(t => sectionTags.includes(t));
        const visible = textMatch && tagMatch;
        s.style.display = visible ? '' : 'none';
        if (tocItems[i]) tocItems[i].style.display = visible ? '' : 'none';
        if (visible) shown += 1;
    });
    const parts = [];
    if (q) parts.push(`text "${input.value}"`);
    if (activeTags.size) parts.push(`tags: ${[...activeTags].join(', ')}`);
    counter.textContent = parts.length
        ? `${shown} of ${total} farm-seasons match (${parts.join(' + ')})`
        : `${total} farm-seasons`;
}

input.addEventListener('input', applyFilter);
tagButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const tag = btn.dataset.tag;
        if (activeTags.has(tag)) { activeTags.delete(tag); btn.classList.remove('active'); }
        else { activeTags.add(tag); btn.classList.add('active'); }
        applyFilter();
    });
});
clearBtn.addEventListener('click', () => {
    activeTags.clear();
    tagButtons.forEach(b => b.classList.remove('active'));
    input.value = '';
    applyFilter();
});
applyFilter();
"""

# Tag filter UI — order matters: positive/quality tags first, then problem flags
TAG_GROUPS = [
    ("Samples", ["5+samples", "2-samples", "3-samples", "4-samples", "1-sample", "0-samples"]),
    ("Harvests", ["2+harvests", "1-harvest", "0-harvests"]),
    ("Line info", ["any-line-specified", "no-line-info"]),
    ("Other", ["multi-species", "has-both", "no-pre-harvest-sample"]),
]
tag_filter_html = '<div class="tag-filters"><strong>Filter by data-quality tag:</strong> '
for group_label, group_tags in TAG_GROUPS:
    tag_filter_html += f'<span style="color:#888;margin:0 4px 0 8px">{group_label}:</span>'
    for tag in group_tags:
        tag_filter_html += f'<button data-tag="{tag}">{tag}</button>'
tag_filter_html += '<button id="clear-tags" class="clear">× clear</button></div>'

definitions_html = """
<details style="background:white;padding:10px 14px;border-radius:6px;margin-bottom:12px;border:1px solid #eee;font-size:13px;">
<summary style="cursor:pointer;font-weight:600;color:#444">Tag definitions</summary>
<div style="margin-top:8px;line-height:1.6">
<b>Sample tags</b> (filter buckets):
<code>5+samples</code> = 5 or more samples · <code>N-samples</code> = exactly N samples (for N = 2, 3, 4) · <code>1-sample</code> = just one · <code>0-samples</code> = none.
The per-panel subtitles show the actual exact count.<br>
<b>Harvest tags</b>:
<code>2+harvests</code> = 2 or more harvests · <code>1-harvest</code> = exactly one · <code>0-harvests</code> = none.<br>
<b>Line info</b>:
<code>any-line-specified</code> = at least one event mentions a specific line in its Notes;
<code>no-line-info</code> = none do. <i>(Note: 144 of 156 farm-seasons have no line info in this dataset.)</i><br>
<b>Other</b>:
<code>multi-species</code> = ≥2 species grown · <code>has-both</code> = has at least 1 sample AND at least 1 harvest (analysis-ready) ·
<code>no-pre-harvest-sample</code> = all samples were taken on or after the first harvest event (unusual ordering).<br>
Multiple tags use AND-logic (a section must have all selected tags to show).
</div>
</details>
"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GreenWave — Farm-Season Plots (interactive)</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>{css}</style></head>
<body>
<header>
  <h1>GreenWave — Farm-Season Plots</h1>
  <div class="summary">
    {len(fs_pairs)} farm-seasons &middot; {len(sh[sh["Log Type"]=="sample"])} sample logs &middot;
    {len(sh[sh["Log Type"]=="harvest"])} harvest logs.<br>
    <b>Hover any point</b> to see exact date, value, and notes. Each panel shows raw sample weights (green dots),
    a daily-averaged growth curve (dark green line — pools same-day replicates), and harvest yields (orange squares).
    All weights are in <b>lb/ft of line</b> — samples come from a 12-inch cut of line; harvests divide total weight by total line length.
    Harvests without a recorded line length appear as dashed vertical orange lines.
  </div>
</header>
<div class="filter-bar">
  <input id="search" placeholder="Filter by farm, season, species…">
  <span class="count" id="match-count"></span>
</div>
{tag_filter_html}
{definitions_html}
<div class="toc"><ul>{''.join(nav_items)}</ul></div>
{''.join(sections)}
<script>{js}</script>
</body></html>
"""

out = Path("/home/claude/farm_season_plots.html")
out.write_text(html)
print(f"Wrote {out} — {out.stat().st_size / 1024:.0f} KB")
