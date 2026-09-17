from scipy.stats import gaussian_kde, lognorm, norm
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from .curves import CURVES

 
NU = 4  # Student-t degrees of freedom (heavy tails: outlier tolerance)

def harvest_scale(draws, c_param=None):
    """
    Multiplicative harvest scale per posterior/prior draw.

    ``c_param`` names the parameterization the caller thinks it is using.
    Only one exists right now (a single multiplicative factor ``c``), so the
    argument is accepted for call-site symmetry and otherwise ignored.
    """
    if c_param not in (None, "c", "harvest_factor"):
        raise ValueError(f"unknown harvest-scale parameterization {c_param!r}")
    return np.asarray(draws["c"], float)
# each parameter gets its own prior distribution
PARAMS = ["A", "k", "t0", "c", "sigma", "sigma_h"]
# parameters to smooth in log space, everything that is positive, non-zero values
LOG_PARAMS = {"A", "k", "c", "sigma", "sigma_h"}


def make_prior_kde(cloud):
    """
    One KDE per parameter from historical fits.
    
    Filter to ok quality fits. Taking all as good right now
    Returns the KDE/prior disstribution for all 6 parameters.
    """
    good = cloud[cloud["quality"] == "ok"]
    prior = {}
    for p in PARAMS: 
        sub = good
        if p == "sigma_h" and "n_harvests" in good:
            # harvest sigma estimate needs 3 harvests at least
            sub = good[good["n_harvests"] >= 3]
        vals = sub[p].dropna().to_numpy()
        if p in LOG_PARAMS: # take log of parameter
            vals = np.log(vals[vals > 0])
        # construct the prior for this parameter. gaussian_kde needs at least
        # two values AND some spread (identical values give a singular
        # covariance), otherwise there is no usable prior here
        usable = len(vals) >= 2 and np.ptp(vals) > 0
        prior[p] = {"kind": "kde",
                    "kde": gaussian_kde(vals) if usable else None,
                    "log": p in LOG_PARAMS}
    prior["_meta"] = {"method": "kde", "source": f"{len(good)} historical fits",
                      "c_param": "c"}
    return prior

def _season_sort_key(season):
    """'Season 23/24' -> 2023 so seasons order chronologically."""
    digits = "".join(ch for ch in str(season) if ch.isdigit())
    return int(digits[:2]) + 2000 if len(digits) >= 2 else -1


 
 
def _spec_usable(spec):
    """A prior spec can be drawn from / handed to PyMC."""
    if spec is None:
        return False
    return spec.get("kind", "kde") != "kde" or spec.get("kde") is not None


def _add_pymc_prior(name, spec, n_grid=400):
    """
    Turn one parameter's prior spec into a PyMC random variable of that name.

    Handles every spec kind the prior builders emit: a "kde" spec (smoothed
    historical fits, optionally in log space) becomes an Interpolated over a
    grid of `n_grid` points; "lognormal"/"normal" specs (the broad population
    prior) map straight onto the matching PyMC distribution.
    """
    kind = spec.get("kind", "kde")
    if kind == "kde":
        kde = spec.get("kde")
        if kde is None:
            raise ValueError(f"no KDE information for {name}")
        lo = kde.dataset.min() - 2 * kde.dataset.std()
        hi = kde.dataset.max() + 2 * kde.dataset.std()
        grid = np.linspace(lo, hi, n_grid)
        pdf = np.clip(kde(grid), 1e-300, None)
        if spec.get("log"):
            # the KDE was fit to log(parameter); exponentiate back
            raw = pm.Interpolated(f"log_{name}", grid, pdf)
            return pm.Deterministic(name, pt.exp(raw))
        return pm.Interpolated(name, grid, pdf)
    if kind == "lognormal":
        return pm.LogNormal(name, mu=spec["mu"], sigma=spec["sigma"])
    if kind == "normal":
        return pm.Normal(name, mu=spec["mu"], sigma=spec["sigma"])
    raise ValueError(f"unknown prior kind {kind!r} for {name}")

def posterior_mcmc(prior, curve_name, t, y, th=None, yh=None, draws=1000, tune=1000, chains=2,
                   target_accept=0.9, rng=0, progressbar=False, c_param=None):
    """
    NUTS posterior for one farm-season.

    ``c_param`` only names the harvest-scale parameterization for validation;
    see `harvest_scale`.
    """
    harvest_scale({"c": np.array([1.0])}, c_param)  # validate the name early
    curve_fn = CURVES[curve_name]
    t, y = np.asarray(t, float), np.asarray(y, float)
    has_h = th is not None and yh is not None and len(np.atleast_1d(th)) > 0
    if has_h:
        th, yh = np.asarray(th, float), np.asarray(yh, float)
    scale_name = "c"

    with pm.Model():
        A = _add_pymc_prior("A", prior["A"])
        k = _add_pymc_prior("k", prior["k"])
        t0 = _add_pymc_prior("t0", prior["t0"])
        sigma = _add_pymc_prior("sigma", prior["sigma"])
        if len(t):
            pm.StudentT("obs_s", nu=NU, mu=curve_fn(t, A, k, t0, m=pt),
                        sigma=sigma, observed=y)
        if has_h:
            scale = _add_pymc_prior(scale_name, prior[scale_name])
            # too few historical harvests to learn a separate harvest noise
            # scale -> reuse the sample-scale sigma
            spec_h = prior.get("sigma_h")
            sigma_h = (_add_pymc_prior("sigma_h", spec_h)
                       if _spec_usable(spec_h) else sigma)
            pm.StudentT("obs_h", nu=NU,
                        mu=scale * curve_fn(th, A, k, t0, m=pt),
                        sigma=sigma_h, observed=yh)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=target_accept, random_seed=rng,
                          progressbar=progressbar, compute_convergence_checks=False)

    post = {}
    for p in ["A", "k", "t0", "sigma", "sigma_h", scale_name]:
        if p in idata.posterior:
            post[p] = idata.posterior[p].values.reshape(-1)
    n = len(post["A"])
    if scale_name not in post or "sigma_h" not in post:
        # no harvests observed -> nothing in the data speaks to these, so
        # they stay at their PRIOR
        # seeded off this call's rng: left unseeded these draws come from
        # fresh OS entropy, which makes every harvest forecast in a season
        # with no prior harvest irreproducible run to run
        pri = sample_prior({k: v for k, v in prior.items()
                            if k in (scale_name, "sigma_h", "_meta")},
                           n=n, rng=rng)
        post.setdefault(scale_name, pri.get(scale_name, np.full(n, np.nan)))
        post.setdefault("sigma_h", pri.get("sigma_h", np.full(n, np.nan)))
    summ = az.summary(idata, var_names=[v for v in ["A", "k", "t0", scale_name]
                                        if v in idata.posterior], kind="diagnostics")
    post["rhat_max"] = float(summ["r_hat"].max())
    post["divergences"] = int(idata.sample_stats.diverging.values.sum())
    post["ess"] = float(summ["ess_bulk"].min())
    post["method"] = "mcmc"
    return post


# Following functions are written by Claude and are work-in-progress.
# Need to be checked by a human.

# Broad population prior: plausible values across ALL farms and seasons,
# chosen from domain knowledge, not fit to this dataset.  SANITY-CHECK the
# ranges against plot_cloud() and adjust — they are config, not truth.
BROAD_PRIOR_CONFIG = {
    # central 95% ranges implied by (mu, sigma) in log space:
    "A":       {"kind": "lognormal", "mu": np.log(4.0),  "sigma": 0.80},  # ~0.8-19 lbs/ft plateau
    "k":       {"kind": "lognormal", "mu": np.log(0.05), "sigma": 0.75},  # ~0.011-0.22 per day
    "t0":      {"kind": "normal",    "mu": 150.0,        "sigma": 45.0},  # inflection late Dec-May
    "c":       {"kind": "lognormal", "mu": np.log(0.65), "sigma": 0.35},  # ~0.33-1.3 harvest/sample
    "sigma":   {"kind": "lognormal", "mu": np.log(0.5),  "sigma": 0.80},  # sample noise lbs/ft
    "sigma_h": {"kind": "lognormal", "mu": np.log(0.5),  "sigma": 0.80},
}

def make_prior_broad(config=None):
    cfg = dict(BROAD_PRIOR_CONFIG)
    if config:
        cfg.update(config)
    params = PARAMS
    prior = {p: dict(cfg[p]) for p in params}
    prior["_meta"] = {"method": "broad", "source": "population (domain ranges)",
                      "c_param": "c"}
    return prior

def _patch_unusable(prior, note="broad"):
    """
    Replace any parameter with no usable prior by its broad population spec.

    A KDE needs >= 2 historical values, so a thin slice of the cloud (one
    farm, or an early season under strict temporal filtering) can leave
    parameters with ``kde=None``.  PyMC cannot build a model from those, so
    fall back to the domain-range prior for exactly those parameters and say
    so in the metadata.
    """
    broad = BROAD_PRIOR_CONFIG
    patched = []
    for param in PARAMS:
        if not _spec_usable(prior.get(param)) and param in broad:
            prior[param] = dict(broad[param])
            patched.append(param)
    if patched:
        prior["_meta"]["source"] += f" (+{note} {'/'.join(patched)})"
    return prior


def make_prior_for_farm(cloud, farm, before_season=None, method="kde",
                        min_seasons=3, fallback="kde",
                        strict_temporal=False, verbose=True):
    """
    Farm-specific prior: built from THIS farm's earlier seasons when it
    has enough of them, falling back to the population otherwise.

    before_season : the season being predicted; only strictly-earlier
        seasons of this farm are used (no peeking at the target season).
    method        : 'kde' (smooth the farm's own fits) or 'broad'
        (skip history entirely; farm arg then only matters via fallback).
    fallback      : 'kde' (population KDE, target season excluded) or
        'broad' when the farm has < min_seasons of usable history.
    strict_temporal : if True the population fallback also drops OTHER
        farms' seasons from the target year onward (no future leakage
        into the backtest).
    """
    if method == "broad":
        prior = make_prior_broad()
        prior["_meta"]["source"] = "population (broad, no history)"
        return prior

    cutoff = _season_sort_key(before_season) if before_season else None
    own = cloud[(cloud["farm"] == farm) & (cloud["quality"] == "ok")]
    if cutoff is not None:
        own = own[own["season"].map(_season_sort_key) < cutoff]

    if len(own) >= min_seasons:
        prior = make_prior_kde(own)
        prior["_meta"]["source"] = (f"farm {farm}: {len(own)} earlier seasons")
        # a single farm rarely has >=3 seasons with >=3 harvests each, so
        # sigma_h / c KDEs can come back None; patch from population
        pop = cloud[cloud["farm"] != farm] if before_season is None else \
              cloud[~((cloud["farm"] == farm) &
                      (cloud["season"].map(_season_sort_key) >= cutoff))]
        pop_prior = make_prior_kde(pop)
        borrowed = []
        for param in PARAMS:
            if not _spec_usable(prior.get(param)) and _spec_usable(pop_prior.get(param)):
                prior[param] = pop_prior[param]
                borrowed.append(param)
        if borrowed:
            prior["_meta"]["source"] += f" (+population {'/'.join(borrowed)})"
        return _patch_unusable(prior)

    # fallback: population, always excluding the target season itself
    pop = cloud
    if before_season is not None:
        pop = pop[~((pop["farm"] == farm) & (pop["season"] == before_season))]
        if strict_temporal:
            pop = pop[pop["season"].map(_season_sort_key) < cutoff]
    if verbose:
        print(f"  {farm}: only {len(own)} earlier seasons "
              f"(< {min_seasons}) -> {fallback} population fallback")
    if fallback == "broad":
        return make_prior_broad()
    prior = make_prior_kde(pop)
    prior["_meta"]["source"] = f"population KDE ({len(pop)} seasons, target excluded)"
    return _patch_unusable(prior)

# ---------------------------------------------------------------------------
# Sampling any prior (drives the fan plots and the no-data panels)
# ---------------------------------------------------------------------------
def sample_prior(prior, n=4000, rng=None):
    """
    Draw `n` samples from every parameter's prior.

    Dispatches on each spec's own ``kind``, so KDE priors and the broad
    population prior (lognormal / normal) both work.  A parameter with no
    usable prior comes back as all-NaN rather than raising: downstream code
    reads that as "no information about this parameter".
    """
    rng = np.random.default_rng(rng)
    out = {}
    for p, spec in prior.items():
        if p == "_meta":
            continue
        kind = spec.get("kind", "kde")
        if kind == "kde":
            if spec.get("kde") is None:
                out[p] = np.full(n, np.nan)
            else:
                v = spec["kde"].resample(n, seed=rng)[0]
                out[p] = np.exp(v) if spec.get("log") else v
        elif kind == "lognormal":
            out[p] = lognorm.rvs(s=spec["sigma"], scale=np.exp(spec["mu"]),
                                 size=n, random_state=rng)
        elif kind == "normal":
            out[p] = norm.rvs(loc=spec["mu"], scale=spec["sigma"],
                              size=n, random_state=rng)
        else:
            raise ValueError(f"unknown prior kind {kind!r} for {p}")
    return out