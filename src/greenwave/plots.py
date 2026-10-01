"""
Visualisation includes plots and tables.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymc as pm
from scipy.stats import gaussian_kde

from .model import (C_PARAM, CURVES, LOG_SCALE_PARAMS, NU,
                    harvest_scale, noise_rng, sample_prior)
from .evaluate import REPORT_GROUPS, REPORT_VIEWS
from .preprocess import (HARVEST_EVENT, IDCOL, OUTPLANT_EVENT, SAMPLE_EVENT,
                         TCOL, YCOL, season_events)

C_PRIOR = "#2a78d6"    # slot 1, blue   - the model
C_FITS = "#eb6834"     # slot 2, orange - sample scale / fitted values
C_ACTUAL = "#1baf7a"   # slot 3, aqua   - what actually happened
INK = "#0b0b0b"
INK_2 = "#52514e"
MIN_SAMPLES = 3


def view(report, columns="compact"):
    """
    A narrower look at a report, still numeric.

    Stays numeric on purpose, so the result can still be rounded and used in arithmetic. Use
    `to_text` only at the point of display.

    Args:
        report (pd.DataFrame): an `evaluate.season_report` frame.
        columns (str | list[str]): a preset name from `REPORT_VIEWS`, a list of group names from
            `REPORT_GROUPS`, or a plain list of column names. Defaults to "compact".

    Returns:
        pd.DataFrame: the selected columns in the order asked for. Columns the report does not have
            are skipped rather than raising, so a view works on a report built without
            `line_events`.
    """
    if isinstance(columns, str):
        columns = REPORT_VIEWS.get(columns, [columns])

    out = []

    for c in columns:
        out.extend(REPORT_GROUPS.get(c, [c]))

    return report[[c for c in out if c in report.columns]]


def to_text(report, nd=2):
    """
    Display version: ranges collapsed to "[lo - hi]", numbers formatted.

    Presentation only. Do arithmetic on the numeric report, never on this, since a column holding
    strings cannot be rounded or summed.

    Args:
        report (pd.DataFrame): a report or a `view` of one.
        nd (int): decimal places. Defaults to 2.

    Returns:
        pd.DataFrame: the same rows with median and range triples merged into formatted strings.
    """
    t = report
    out = pd.DataFrame(index=t.index)
    pairs = {}

    for c in t.columns:
        if c.endswith("_lo95") or c.endswith("_lo95_after"):
            hi = c.replace("_lo95", "_hi95")
            if hi in t.columns:
                pairs[c] = hi

    skip = set(pairs) | set(pairs.values())

    for c in t.columns:
        if c in skip:
            continue
        if pd.api.types.is_float_dtype(t[c]):
            big = t[c].abs().max()
            d = 0 if (pd.notna(big) and big >= 1000) else nd
            out[c] = [("" if not np.isfinite(v) else f"{v:,.{d}f}") for v in t[c]]
        else:
            out[c] = t[c].astype(object).where(t[c].notna(), "")

    for lo, hi in pairs.items():
        big = t[[lo, hi]].abs().max().max()
        d = 0 if (pd.notna(big) and big >= 1000) else nd
        out[lo.replace("_lo95", "_95")] = [
            "" if not np.isfinite(a) else f"[{a:,.{d}f} – {b:,.{d}f}]"
            for a, b in zip(t[lo], t[hi])]

    return out


def add_biomass_in_water(report, line_events=None):
    """
    Pounds still standing: the harvest forecast times feet in the water.

    This is the app's number. Two things move it, the lbs/ft forecast as observations arrive and the
    footage as harvests remove line, so it is a running quantity that applies to every row rather
    than a per-harvest one.

    Both scales are added side by side. The gap between the sample-scale and harvest-scale pounds is
    the harvest/sample factor `c`, in pounds.

    Args:
        report (pd.DataFrame): an `evaluate.season_report` frame. Needs `line_in_water_ft`, so the
            report must have been built with `line_events`.
        line_events (pd.DataFrame | None): accepted for call-site symmetry; the footage is read off
            the report.

    Returns:
        pd.DataFrame: the report plus the in-water pounds on both scales.
    """
    t = report.copy()

    if "line_in_water_ft" not in t.columns or t["line_in_water_ft"].isna().all():
        raise ValueError("no line_in_water_ft; pass line_events= to season_report")

    ft = t["line_in_water_ft"]
    t["lbs_in_water_med"] = t["pred_harvest_lbs_ft"] * ft
    t["lbs_in_water_lo95"] = t["harvest_yield_lo95"] * ft
    t["lbs_in_water_hi95"] = t["harvest_yield_hi95"] * ft

    if "sample_yield_lo95" in t.columns:
        t["lbs_in_water_sample_med"] = t["pred_sample_lbs_ft"] * ft
        t["lbs_in_water_sample_lo95"] = t["sample_yield_lo95"] * ft
        t["lbs_in_water_sample_hi95"] = t["sample_yield_hi95"] * ft

    return t


def add_harvest_pounds(report):
    """
    Pounds for the line a harvest actually took. The evaluation number.

    Harvest rows only, using `line_harvested_ft`, which is knowable only after the fact. Using
    feet-in-the-water here would compare a prediction for the whole farm against a weight from part
    of it.

    The reading range is carried through so `in95_reading` can be read in pounds, though the verdict
    is identical either way: scaling both ends and the actual by the same footage cannot change
    containment.

    Args:
        report (pd.DataFrame): an `evaluate.season_report` frame.

    Returns:
        pd.DataFrame: the report plus the predicted pounds, ranges, and error. Non-harvest rows are
            NaN in those columns.
    """
    t = report.copy()
    is_h = t["event"] == HARVEST_EVENT
    ft = t["line_harvested_ft"].where(is_h)
    t["pred_lbs_med"] = t["pred_harvest_lbs_ft"] * ft
    t["pred_lbs_lo95"] = t["harvest_yield_lo95"] * ft
    t["pred_lbs_hi95"] = t["harvest_yield_hi95"] * ft
    t["pred_lbs_reading_lo95"] = t["harvest_reading_lo95"] * ft
    t["pred_lbs_reading_hi95"] = t["harvest_reading_hi95"] * ft
    t["err_lbs"] = t["pred_lbs_med"] - t["actual_lbs"]

    return t


def plot_season_curves(model, fsid, as_of=None, n_obs=None, estimator="median",
                       t_max=None, show_sample_curve=True, savepath=None,
                       ax=None):
    """
    One posterior, drawn as curves across the whole season.

    The "what did the model believe, and was it right" picture. It shows the 95% band and median for
    harvest yield across all days, labelled the farm's average that day; the wider band for any
    single cut; a dashed sample-scale median; observations the posterior was conditioned on filled
    and ones it was not hollow; and a dotted line at the cutoff.

    This is one posterior extended across time, which is what makes the hollow markers meaningful:
    they are a genuine out-of-sample check rather than a different fit's prediction.

    Outplants are drawn as brown squares at y = 0 and are always solid, since they are never
    conditioned on by any posterior, so the filled or hollow distinction does not apply to them.

    Args:
        model (SeasonModel): supplies the events, prior, and posteriors.
        fsid (str): farm-season id.
        as_of (int | float | str | pd.Timestamp | None): conditioning cutoff. Exactly one of this
            and `n_obs`; `n_obs=0` draws the prior alone.
        n_obs (int | None): condition on a count of observations instead.
        estimator (str): accepted for symmetry; the bands are always quantiles of the draws.
        t_max (float | None): last day drawn. Defaults to 25 days past the last observation, so the
            extrapolation is visible.
        show_sample_curve (bool): draw the dashed sample-scale median. Defaults to True.
        savepath (str | None): write the figure here.
        ax (matplotlib.axes.Axes | None): draw into an existing axes instead of making a figure.

    Returns:
        matplotlib.figure.Figure | None: the figure, or None when `ax` was given.
    """
    import matplotlib.pyplot as plt

    c_actual, ink_2 = "#1baf7a", "#52514e"
    obs = season_events(model.events, fsid)

    if not obs:
        raise ValueError(f"no observations for {fsid}")

    post = model.posterior(fsid, as_of=as_of, n_obs=n_obs)
    n_seen = post["_n_seen"]
    cutoff = post["_last_day"] if n_seen else (obs[0][0] - 1)
    meta = model.events[model.events[IDCOL] == fsid].iloc[0]

    f = CURVES[model.cache.curve]
    last = max(d for d, *_ in obs)
    grid = np.arange(0, (t_max or last + 25) + 1, 2.0)
    curves = harvest_scale(post, C_PARAM)[:, None] * f(
        grid[None, :], post["A"][:, None], post["k"][:, None], post["t0"][:, None])
    lo, med, hi = np.quantile(curves, [0.025, 0.5, 0.975], axis=0)
    sig_h = np.asarray(post.get("sigma_h", np.nan), float)[:, None]
    rng = noise_rng(fsid, n_seen, "curveplot")
    noisy = np.clip(curves + sig_h * rng.standard_t(NU, size=curves.shape), 0, None)
    plo, phi = np.quantile(noisy, [0.025, 0.975], axis=0)

    fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))

    ax.fill_between(grid, plo, phi, color=C_PRIOR, alpha=0.10, lw=0,
                    label="95% — any single cut")
    ax.fill_between(grid, lo, hi, color=C_PRIOR, alpha=0.24, lw=0,
                    label="95% — the farm's average that day")
    ax.plot(grid, med, color=C_PRIOR, lw=2, label="harvest forecast (median)")

    if show_sample_curve:
        s_med = np.median(f(grid[None, :], post["A"][:, None], post["k"][:, None],
                            post["t0"][:, None]), axis=0)
        ax.plot(grid, s_med, color=C_FITS, lw=1.4, ls="--",
                label="what a 1-ft sample would read (median)")

    op = model.events[(model.events[IDCOL] == fsid)
                      & (model.events.event == OUTPLANT_EVENT)][TCOL].astype(float)

    if len(op):
        ax.scatter(op, np.zeros(len(op)), marker="s", s=38, color="saddlebrown",
                   zorder=4, label="outplant")

    for kind, marker, col in [(SAMPLE_EVENT, "o", C_FITS),
                              (HARVEST_EVENT, "*", c_actual)]:
        d = np.array([e[0] for e in obs if e[2] == kind])
        v = np.array([e[1] for e in obs if e[2] == kind])
        if not len(d):
            continue
        seen = d <= cutoff
        ms = 110 if kind == HARVEST_EVENT else 34
        ax.scatter(d[seen], v[seen], marker=marker, s=ms, color=col,
                   edgecolor="white", linewidth=0.6, zorder=5,
                   label=f"{kind} (conditioned on)")
        if (~seen).any():
            ax.scatter(d[~seen], v[~seen], marker=marker, s=ms, facecolor="none",
                       edgecolor=col, linewidth=1.4, zorder=5,
                       label=f"{kind} (NOT seen by this fit)")

    if n_seen:
        ax.axvline(cutoff, color=ink_2, ls=":", lw=1.2)
        ax.text(cutoff, ax.get_ylim()[1] * 0.98, " conditioned up to here",
                fontsize=8, color=ink_2, va="top")

    ymax = max(max(v for _, v, *_ in obs), float(np.nanmax(hi)))
    ax.set_ylim(0, 1.25 * ymax)
    ax.set_xlabel("day of season", color=ink_2)
    ax.set_ylabel("lbs/ft", color=INK)
    ax.set_title(f"{meta['Farm Name']} {meta['Season']} — one posterior, "
                 f"conditioned on {n_seen} of {len(obs)} observations "
                 f"(through day {cutoff:.0f})", fontsize=10, color=INK, loc="left")

    # the two bands answer different questions, and the labels alone do not
    # make that obvious -- so say it on the figure
    ax.text(0, 1.015, "inner band: where the farm's average yield per foot sits."
                      "   outer band: where any one cut can land, which is wider "
                      "because individual stretches scatter around that average.",
            transform=ax.transAxes, fontsize=7.5, color=ink_2, va="bottom")

    ax.grid(alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(fontsize=7.5, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5))

    if fig is not None:
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"wrote {savepath}")

    return fig, ax


def plot_fit(df, cloud, fsid=None, farm=None, season=None, curve="logistic",
             ax=None, legend=True, small=False, savepath=None):
    """
    One farm-season: its observations and its least-squares fitted curve.

    A data-inspection panel, not a model output. The curve comes from
    `fit_curve.build_parameter_cloud`, not from a posterior.

    Args:
        df (pd.DataFrame): events table with outplants, samples, and harvests.
        cloud (pd.DataFrame): `fit_curve.build_parameter_cloud` output.
        fsid (str | None): farm-season id. Give this or `farm` plus `season`.
        farm (str | None): matched case-insensitively as a substring, so a short name is enough.
        season (str | None): season label, used with `farm`.
        curve (str): key into `model.CURVES`.
        ax (matplotlib.axes.Axes | None): draw into an existing axes; `plot_fits` tiles panels this
            way.
        legend (bool): draw the legend. Defaults to True.
        small (bool): compact fonts and markers, for tiled panels.
        savepath (str | None): write the figure here.

    Returns:
        matplotlib.figure.Figure | None: the figure, or None when `ax` was given.
    """
    f = CURVES[curve]

    if fsid is not None:
        hit = cloud[cloud[IDCOL] == fsid]
    else:
        hit = cloud
        if farm is not None:
            hit = hit[hit["farm"].str.contains(farm, case=False, na=False)]
        if season is not None:
            hit = hit[hit["season"] == season]

    if len(hit) == 0:
        raise ValueError(f"no farm-season matched "
                         f"(fsid={fsid!r}, farm={farm!r}, season={season!r})")

    if len(hit) > 1:
        raise ValueError(f"{len(hit)} farm-seasons matched; be more specific: "
                         + ", ".join(hit[IDCOL].astype(str).head(6)))

    row = hit.iloc[0]
    fsid = row[IDCOL]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    g = df[(df[IDCOL] == fsid) & (df["event"] == SAMPLE_EVENT)]
    h = df[(df[IDCOL] == fsid) & (df["event"] == HARVEST_EVENT)]
    o = df[(df[IDCOL] == fsid) & (df["is_outplant"] == True)]  # noqa: E712

    # Outplant events as green squares at y=0 (biomass starts at ~zero):
    # shows where the season began and, for multi-outplant farms, how
    # staggered the plantings were.
    odays = sorted(o[TCOL].dropna().unique())

    if odays:
        ax.scatter(odays, np.zeros(len(odays)), marker="s", s=45,
                   color="tab:green", zorder=4, label="outplant")

    ax.scatter(g[TCOL], g[YCOL], s=18, color="tab:blue", zorder=3,
               label="samples")

    if len(h):
        ax.scatter(h[TCOL], h[YCOL], marker="*", s=120,
                   color="tab:orange", zorder=4, label="harvest")

    # Extend the curve well past the last sample so extrapolated
    # plateaus (unidentified A) are visually obvious; start it early
    # enough to cover the first outplant.
    t_end = max(g[TCOL].max(), h[TCOL].max() if len(h) else 0) + 30
    t_start = min(g[TCOL].min(),
                  o[TCOL].min() if len(o) else g[TCOL].min(), 0)
    tt = np.linspace(t_start, t_end, 200)

    if np.isfinite(row["A"]):
        yy = f(tt, row["A"], row["k"], row["t0"])
        ax.plot(tt, yy, color="tab:red", lw=1.5, zorder=2, label="fit")
        ax.axhline(row["A"], color="tab:red", ls=":", lw=0.8, alpha=0.6,
                   label="A (plateau)")
        # Harvest-scale curve c*f(t): what the fit predicts a HARVEST
        # measurement would read — the stars should track this line.
        if np.isfinite(row.get("c", np.nan)):
            ax.plot(tt, row["c"] * yy, color="tab:orange", lw=1.2,
                    ls="--", zorder=2, label="c x fit (harvest scale)")

    tag = "" if row["quality"] == "ok" else "  ⚠"
    c_txt = (f" c={row['c']:.2f}"
             if np.isfinite(row.get("c", np.nan)) else "")
    ts, ls = (8, 7) if small else (10, 9)
    ax.set_title(f"{row['farm'][:18] if small else row['farm']} "
                 f"{row['season']}{tag}\n"
                 f"A={row['A']:.2f} k={row['k']:.3f} "
                 f"t0={row['t0']:.0f}{c_txt} n={row['n_samples']}",
                 fontsize=ts)

    ax.set_xlabel("day of season", fontsize=ls)
    ax.set_ylabel(YCOL, fontsize=ls)
    ax.tick_params(labelsize=ls)

    if legend:
        ax.legend(fontsize=8, frameon=False)

    if savepath:
        ax.figure.savefig(savepath, dpi=150, bbox_inches="tight")

    return ax


def plot_fits(df, cloud, curve="logistic", ncols=4, max_plots=None,
              savepath=None):
    """
    A grid of fitted curves, one panel per farm-season.

    Sorted with the worst rmse first, since the point is to see where the logistic shape does not
    hold rather than to admire the good fits. Every fit is drawn; there is deliberately no filter on
    `quality`, because that flagging is provisional and has never been validated, so a flagged fit
    is marked rather than hidden.

    Args:
        df (pd.DataFrame): events table with outplants, samples, and harvests.
        cloud (pd.DataFrame): `fit_curve.build_parameter_cloud` output.
        curve (str): key into `model.CURVES`.
        ncols (int): panels per row. Defaults to 4.
        max_plots (int | None): cap the number of panels. None draws all 126.
        savepath (str | None): write the figure here.

    Returns:
        matplotlib.figure.Figure: the tiled figure.
    """
    plot_cloud = cloud if max_plots is None else cloud.head(max_plots)
    n = len(plot_cloud)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for ax, (_, row) in zip(axes.flat, plot_cloud.iterrows()):
        plot_fit(df, cloud, fsid=row[IDCOL], curve=curve, ax=ax,
                 legend=False, small=True)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Per-season {curve} fits (sorted worst RMSE first; "
                 f"⚠ = flagged)", y=1.001)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_cloud(cloud, savepath=None):
    """
    Pairwise scatter of the fitted parameters. Each dot is one farm-season.

    Shows the shape the priors have to cover, and the ridges between parameters, `A` against `t0`
    especially, that a set of independent marginal priors cannot express.

    Flagged fits are grey and unflagged ones blue. The flags are provisional; they are shown so a
    cluster can be recognised as a fitting artefact rather than a real population.

    Args:
        cloud (pd.DataFrame): `fit_curve.build_parameter_cloud` output.
        savepath (str | None): write the figure here.

    Returns:
        matplotlib.figure.Figure: the scatter grid.
    """
    ok = cloud[cloud["quality"] == "ok"]
    bad = cloud[cloud["quality"] != "ok"]
    has_c = "c" in cloud.columns and cloud["c"].notna().any()
    pairs = [("t0", "A"), ("t0", "k"), ("k", "A")]

    if has_c:
        pairs += [("c", "A"), ("c", "sigma_h")]

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs), 4))

    for ax, (x, y) in zip(np.atleast_1d(axes), pairs):
        ax.scatter(bad[x], bad[y], s=25, color="lightgray", label="flagged")
        ax.scatter(ok[x], ok[y], s=25, color="tab:blue", label="ok")
        ax.set_xlabel(x); ax.set_ylabel(y)
        if x == "c":
            ax.axvline(0.65, color="tab:orange", ls=":", lw=1,
                       label="c=0.65" if y == "A" else None)

    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Parameter cloud")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_prior(define_priors, cloud=None, n=20000):
    """
    Each prior against the fitted values it is meant to cover.

    Args:
        define_priors (callable): a prior builder, e.g. `prior_for()`.
        cloud (pd.DataFrame | None): the fitted values to compare against. None draws the priors
            alone.
        n (int): draws used for the prior ranges. Defaults to 20000.

    Returns:
        pd.DataFrame: `prior_coverage`'s table. The figure is shown as a side effect.
    """
    draws = sample_prior(define_priors, n)
    params = [p for p in draws if np.isfinite(draws[p]).any()]
    fig, axes = plt.subplots(1, len(params), figsize=(3.6 * len(params), 3))

    for ax, p in zip(np.atleast_1d(axes), params):
        v = draws[p][np.isfinite(draws[p])]
        logscale = p in LOG_SCALE_PARAMS and (v > 0).all()
        if logscale:
            grid = np.logspace(np.log10(v.min()), np.log10(v.max()), 300)
            ax.plot(grid, gaussian_kde(np.log(v))(np.log(grid)),
                    color="tab:purple", lw=2)
            ax.set_xscale("log")
        else:
            grid = np.linspace(v.min(), v.max(), 300)
            ax.plot(grid, gaussian_kde(v)(grid), color="tab:purple", lw=2)
        if cloud is not None and p in cloud:
            hv = cloud[p].dropna()
            ax.plot(hv, np.zeros(len(hv)), "|", ms=18, color="tab:blue")
        ax.set_title(f"prior for {p}", fontsize=9)
        ax.set_yticks([])

    fig.suptitle("prior marginals", y=1.04)
    fig.tight_layout()
    plt.show(); plt.close(fig)


def plot_prior_curves(define_priors, curve_name, n_curves=100, t_max=280):
    """
    The prior as yield(date) curves.

    Args:
        define_priors (callable): a prior builder.
        curve_name (str): key into `model.CURVES`.
        n_curves (int): curves drawn. Defaults to 100.
        t_max (int): last day of season drawn. Defaults to 280.

    Returns:
        matplotlib.figure.Figure: sample-scale curves in red, harvest-scale dashed orange.
    """
    f = CURVES[curve_name]
    t = np.linspace(0, t_max, 200)
    d = sample_prior(define_priors, n=n_curves)
    sc = harvest_scale(d)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(n_curves):
        y = f(t, d["A"][i], d["k"][i], d["t0"][i])
        ax.plot(t, y, color="tab:red", alpha=0.12, lw=1)
        if np.isfinite(sc[i]):
            ax.plot(t, sc[i] * y, color="tab:orange", alpha=0.12, lw=1, ls="--")

    ax.set_xlabel("day of season"); ax.set_ylabel("lbs_ft")
    ax.set_title(f"{n_curves} seasons drawn from the prior\n"
                 "(red = sample scale, dashed = harvest scale)")
    fig.tight_layout(); plt.show(); plt.close(fig)


def _build(define_priors):
    """
    The prior RVs, created in a throwaway model so names cannot collide.

    Args:
        define_priors (callable): a prior builder.

    Returns:
        dict: `param -> pm RV`, outside any live model context.
    """
    with pm.Model():
        return define_priors()


def _prior_pdf(rv, x):
    """
    Prior density at parameter values `x`, for any pm distribution.

    Args:
        rv: a PyMC random variable.
        x (np.ndarray): parameter values to evaluate at, in natural units.

    Returns:
        np.ndarray: density at each `x`, as probability per unit of the parameter, the same units a
            density histogram of fits uses. That is what lets the two be overlaid honestly.
    """
    return np.exp(pm.logp(rv, np.asarray(x, float)).eval())


def _prior_interval(rv, q=(0.025, 0.975), n=40000, seed=0):
    """
    Central range of one prior, in parameter units, by sampling.

    Args:
        rv: a PyMC random variable.
        q (tuple[float, float]): the quantiles. Defaults to (0.025, 0.975).
        n (int): draws. Defaults to 40000.
        seed (int): seed. Defaults to 0, so a reported range is stable.

    Returns:
        tuple[float, float]: the range.
    """
    d = np.asarray(pm.draw(rv, n, random_seed=seed)).reshape(-1)

    return tuple(np.quantile(d, q))


def prior_coverage(define_priors, cloud, params=None, q=(0.025, 0.975),
                   one_sided=("sigma", "sigma_h")):
    """
    Does each prior actually cover the historical fits?

    `pct_inside` is the share of fitted values the prior admits.


    Args:
        define_priors (callable): a prior builder.
        cloud (pd.DataFrame): `fit_curve.build_parameter_cloud` output.
        params (list[str] | None): which parameters. Defaults to those in both the prior and cloud.
        q (tuple[float, float]): the prior quantiles. Defaults to (0.025, 0.975).
        one_sided (tuple[str, ...]): parameters scored as below the upper quantile. Defaults to the
            two noise scales.

    Returns:
        pd.DataFrame: one row per parameter with the prior's range, the fits' range, `pct_inside`,
            and how the range was read.
    """
    pri = _build(define_priors)
    rows = []

    for p in (params or ["A", "k", "t0", "c", "sigma", "sigma_h"]):
        if p not in pri:
            continue
        vals = (cloud[p].dropna().to_numpy()
                if p in cloud.columns else np.array([]))
        lo, hi = _prior_interval(pri[p], q)
        sided = p in one_sided
        inside = (100 * np.mean(vals <= hi) if sided
                  else 100 * np.mean((vals >= lo) & (vals <= hi))) \
            if len(vals) else np.nan
        rows.append(dict(
            param=p, kind=pri[p].owner.op.name,
            prior_lo=(0.0 if sided else lo), prior_hi=hi,
            interval="one-sided" if sided else "central",
            n_fits=len(vals),
            fits_min=vals.min() if len(vals) else np.nan,
            fits_med=np.median(vals) if len(vals) else np.nan,
            fits_max=vals.max() if len(vals) else np.nan,
            pct_inside=inside))

    return pd.DataFrame(rows)


def plot_prior_marginals(define_priors, cloud=None, params=None, ncols=3,
                         n_grid=500, q=(0.025, 0.975), window="prior",
                         one_sided=("sigma", "sigma_h"), savepath=None):
    """
    One panel per parameter: the prior density against the historical fits.

    Args:
        define_priors (callable): a prior builder.
        cloud (pd.DataFrame | None): fitted values to overlay. None draws the priors alone.
        params (list[str] | None): which parameters.
        ncols (int): panels per row. Defaults to 3.
        n_grid (int): points in each pdf curve. Defaults to 500.
        q (tuple[float, float]): the prior range reported. Defaults to (0.025, 0.975).
        window (str): what the x axis must contain. "prior" is the prior's own range unioned with
            the middle half of the fits, which keeps the prior curve legible and counts outliers
            rather than drawing them. "all" is every fitted value however extreme, so you can see
            how far an uninformed prior sits from the data, at the cost of legibility when the fits
            have a defect tail.
        one_sided (tuple[str, ...]): passed to `prior_coverage`.
        savepath (str | None): write the figure here.

    Returns:
        matplotlib.figure.Figure: the panel grid.
    """
    if window not in ("prior", "all"):
        raise ValueError(f"window must be 'prior' or 'all', got {window!r}")

    pri = _build(define_priors)
    params = params or [p for p in ("A", "k", "t0", "c", "sigma", "sigma_h")
                        if p in pri]
    ok = cloud # keep fits of all qualities 
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.1 * nrows),
                             squeeze=False)

    for ax, p in zip(axes.flat, params):
        rv = pri[p]
        lo, hi = _prior_interval(rv, q)
        vals = (ok[p].dropna().to_numpy()
                if ok is not None and p in ok.columns else np.array([]))
        logx = p in LOG_SCALE_PARAMS and np.isfinite(lo) and lo > 0

        shown = vals[vals > 0] if logx else vals
        n_nonpos = len(vals) - len(shown)

        span = []
        if np.isfinite(lo) and np.isfinite(hi):
            if logx:
                w = np.log10(hi) - np.log10(lo)
                span += [10 ** (np.log10(lo) - 0.5 * w),
                         10 ** (np.log10(hi) + 0.5 * w)]
            else:
                w = hi - lo
                span += [lo - 0.5 * w, hi + 0.5 * w]
        if len(shown):
            span += ([shown.min(), shown.max()] if window == "all" else
                     [np.quantile(shown, 0.25), np.quantile(shown, 0.75)])
        x_lo, x_hi = min(span), max(span)
        if logx:
            grid = np.logspace(np.log10(max(x_lo, 1e-300)),
                               np.log10(x_hi), n_grid)
        else:
            pad = 0.04 * ((x_hi - x_lo) or 1.0)
            grid = np.linspace(x_lo - pad, x_hi + pad, n_grid)

        dens = _prior_pdf(rv, grid)

        inwin = shown[(shown >= grid[0]) & (shown <= grid[-1])]
        n_off = len(shown) - len(inwin) + n_nonpos
        heights, centers = np.array([]), np.array([])
        if len(inwin):
            bins = (np.logspace(np.log10(grid[0]), np.log10(grid[-1]), 24)
                    if logx else np.linspace(grid[0], grid[-1], 24))
            heights, _, _ = ax.hist(inwin, bins=bins, density=True,
                                    color=C_FITS, alpha=0.30, edgecolor="none",
                                    label="historical fits", zorder=2)
            centers = (np.sqrt(bins[:-1] * bins[1:]) if logx
                       else 0.5 * (bins[:-1] + bins[1:]))
            # rug: exact values, since 24 bins hide where the tails really sit
            ax.plot(inwin, np.zeros(len(inwin)), "|", ms=9, color=C_FITS,
                    alpha=0.85, zorder=4)

        band = ((grid <= hi) if p in one_sided
                else ((grid >= lo) & (grid <= hi)))
        ax.fill_between(grid[band], 0, dens[band], color=C_PRIOR, alpha=0.13,
                        zorder=1, label=f"prior central {100*(q[1]-q[0]):.0f}%")
        ax.plot(grid, dens, color=C_PRIOR, lw=2, zorder=3, label="prior")

        if logx:
            ax.set_xscale("log")
        ax.set_xlim(grid[0], grid[-1])

        n_clip = 0
        if window == "all" and len(heights):
            inband = heights[(centers >= lo) & (centers <= hi)]
            y_top = 1.30 * max(np.nanmax(dens),
                               inband.max() if len(inband) else 0.0)
            n_clip = int((heights > y_top).sum())
            ax.set_ylim(0, y_top)
        else:
            ax.set_ylim(bottom=0)

        sided = p in one_sided
        inside = (100 * np.mean(vals <= hi) if sided
                  else 100 * np.mean((vals >= lo) & (vals <= hi))) \
            if len(vals) else np.nan
        ax.set_title(f"{p}  ({rv.owner.op.name})", fontsize=10,
                     color=INK, loc="left")
        rng = ((f"prior {100*q[1]:.0f}th pct: {hi:.3g}" if sided else
                f"prior {100*(q[1]-q[0]):.0f}%: {lo:.3g} – {hi:.3g}")
               + (f"\nfits: {vals.min():.3g} – {vals.max():.3g}"
                  f"\n{inside:.0f}% of {len(vals)} fits inside"
                  if len(vals) else "\n(no fitted values)")
               + (f"\n{n_off} off-scale" if n_off else "")
               + (f"\n{n_clip} bars clipped" if n_clip else ""))
        ax.text(0.97, 0.94, rng, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK_2)
        ax.set_ylabel("density", fontsize=8, color=INK_2)
        ax.tick_params(labelsize=8, colors=INK_2)
        ax.grid(axis="y", color="#e6e5e1", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d2")

    for ax in axes.flat[len(params):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Prior marginals vs historical fits", fontsize=11, color=INK)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.show(); plt.close(fig)


