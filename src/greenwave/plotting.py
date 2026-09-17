import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .curves import CURVES
from .bayesian import sample_prior, LOG_PARAMS, PARAMS, harvest_scale

YCOL = "lbs_ft"
TCOL = "day_of_season"
IDCOL = "farm_season_id"
SAMPLE_EVENT = "sample"      # value in `event` column for samples
OUTPLANT_EVENT = "outplant"  # value in `event` column for outplanting
HARVEST_EVENT = "harvest"    # value in `event` column for harvests
MIN_SAMPLES = 3              # skip seasons with fewer sample points than this

def plot_fits(df, cloud, curve="logistic", ncols=4, max_plots=None,
              savepath=None):
    """
    Plots the fitted curves per farm-season against their observations.

    Uses the "cloud" that records each farm-seasons curve parameters.
    The cloud is sorted with worse RMSE first, max_plots cap how many season panels are drawn.
    """
    f = CURVES[curve]
    samples = df[df["event"] == SAMPLE_EVENT]
    harvests = df[df["event"] == HARVEST_EVENT]
    outplants = df[df["is_outplant"] == True]  # noqa: E712
 
    plot_cloud = cloud if max_plots is None else cloud.head(max_plots)
    n = len(plot_cloud)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
 
    for ax, (_, row) in zip(axes.flat, plot_cloud.iterrows()):
        fsid = row[IDCOL]
        g = samples[samples[IDCOL] == fsid]
        h = harvests[harvests[IDCOL] == fsid]
        o = outplants[outplants[IDCOL] == fsid]
 
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
            ax.plot(tt, yy, color="tab:red", lw=1.5, zorder=2)
            ax.axhline(row["A"], color="tab:red", ls=":", lw=0.8, alpha=0.6)
            # Harvest-scale curve c*f(t): what the fit predicts a HARVEST
            # measurement would read — the stars should track this line.
            if np.isfinite(row.get("c", np.nan)):
                ax.plot(tt, row["c"] * yy, color="tab:orange", lw=1.2,
                        ls="--", zorder=2)
 
        tag = "" if row["quality"] == "ok" else "  ⚠"
        c_txt = (f" c={row['c']:.2f}"
                 if np.isfinite(row.get("c", np.nan)) else "")
        ax.set_title(f"{row['farm'][:18]} {row['season']}{tag}\n"
                     f"A={row['A']:.2f} k={row['k']:.3f} "
                     f"t0={row['t0']:.0f}{c_txt} n={row['n_samples']}",
                     fontsize=8)
        ax.set_xlabel("day of season", fontsize=7)
        ax.set_ylabel(YCOL, fontsize=7)
        ax.tick_params(labelsize=7)
 
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
    Plots the cloud of fitted parameters. Each dot is a farm-season.
    KDE prior will smooth this distribution.

    Marks "bad" points as light gray and blue as non-flagged points.
    Can modify what these flags are.
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


def plot_prior(prior, cloud=None, n=20000):
    """
    Plot prior.
    """
    draws = sample_prior(prior, n)
    params = [p for p in draws if np.isfinite(draws[p]).any()]
    fig, axes = plt.subplots(1, len(params), figsize=(3.6 * len(params), 3))
    for ax, p in zip(np.atleast_1d(axes), params):
        v = draws[p][np.isfinite(draws[p])]
        logscale = p in LOG_PARAMS and (v > 0).all()
        if logscale:
            grid = np.logspace(np.log10(v.min()), np.log10(v.max()), 300)
            ax.plot(grid, gaussian_kde(np.log(v))(np.log(grid)),
                    color="tab:purple", lw=2)
            ax.set_xscale("log")
        else:
            grid = np.linspace(v.min(), v.max(), 300)
            ax.plot(grid, gaussian_kde(v)(grid), color="tab:purple", lw=2)
        if cloud is not None and p in cloud:
            hv = cloud.loc[cloud["quality"] == "ok", p].dropna()
            ax.plot(hv, np.zeros(len(hv)), "|", ms=18, color="tab:blue")
        ax.set_title(f"prior for {p}", fontsize=9)
        ax.set_yticks([])
    fig.suptitle(f"prior marginals  [{prior['_meta']['method']}: "
                 f"{prior['_meta']['source']}]", y=1.04)
    fig.tight_layout()
    plt.show(); plt.close(fig)


def plot_prior_curves(prior, curve_name, n_curves=100, t_max=280):
    """
    The prior as yield(date) curves.

    red = sample scale, dashed orange = harvest
    t_max is the last day of season the curve gets drawn out to
    """
    f = CURVES[curve_name]
    t = np.linspace(0, t_max, 200)
    d = sample_prior(prior, n=n_curves)
    sc = harvest_scale(d)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_curves):
        y = f(t, d["A"][i], d["k"][i], d["t0"][i])
        ax.plot(t, y, color="tab:red", alpha=0.12, lw=1)
        if np.isfinite(sc[i]):
            ax.plot(t, sc[i] * y, color="tab:orange", alpha=0.12, lw=1, ls="--")
    ax.set_xlabel("day of season"); ax.set_ylabel("lbs_ft")
    ax.set_title(f"{n_curves} seasons drawn from the prior "
                 f"[{prior['_meta']['method']}]\n"
                 "(red = sample scale, dashed = harvest scale)")
    fig.tight_layout(); plt.show(); plt.close(fig)