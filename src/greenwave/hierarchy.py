"""
The hierarchical (partially pooled) prior: learn from earlier seasons.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from . import preprocess as pre
from .model import (BROAD_PRIOR, NU, Prior, broad_prior_fingerprint,
                    logistic, make_prior_for_season)

#the hierarchy's parameters, on the scale it works in
ORDER = ["logA", "logk", "t0", "logc", "logsh"]

#: which hierarchy parameter comes from which entry of the broad prior
_FROM_BROAD_PRIOR = {"logA": "A", "logk": "k", "t0": "t0", "logc": "c", "logsh": "sigma_h"}

#: which levels each parameter varies by
CONFIG = {"logA": ["state"], "logk": ["state"], "t0": ["state"],
          "logc": ["farm"], "logsh": []}

MIN_FARMS_PER_REGION = 3
# how much of the broad prior's spread to allow BETWEEN levels
LEVEL_SD_FRACTION = 0.75
STAGE1_MCMC = dict(draws=1000, tune=1000, chains=4, cores=1, target_accept=0.95)
#: stage-1 variables to keep when caching the stage 1 posteriors
_KEEP_PREFIXES = ("mu_", "sd_", "z_state_", "z_farm_")


def hyperprior(broad_prior=None):
    """
    Center and spread per hierarchy parameter, read off the broad prior.

    The centers become the `mu_*` hyperpriors; the spreads set their widths and, scaled
    by `LEVEL_SD_FRACTION`, how far levels may differ.

    Args:
        broad_prior (dict | None): `param -> (family, a, b)`. Defaults to `BROAD_PRIOR`.

    Returns:
        tuple[dict, dict]: `(center, spread)` keyed by the names in `ORDER`.
    """
    broad_prior = BROAD_PRIOR if broad_prior is None else broad_prior
    center = {p: float(broad_prior[_FROM_BROAD_PRIOR[p]][1]) for p in ORDER}
    spread = {p: float(broad_prior[_FROM_BROAD_PRIOR[p]][2]) for p in ORDER}

    return center, spread


def sigma_prior(broad_prior=None):
    """
    The sample-noise prior, which stays out of the hierarchy.

    `sigma` is identical in every fit because the data pins it: a season with 19 samples determines
    it tightly, so borrowing across farms buys nothing. `sigma_h` is in the hierarchy for the
    opposite reason, since two harvests say almost nothing about harvest noise. A deliberate
    asymmetry, not an omission.

    Args:
        broad_prior (dict | None): `param -> (family, a, b)`. Defaults to `BROAD_PRIOR`.

    Returns:
        tuple[float, float]: `(mu, sigma)` for a `pm.LogNormal`.
    """
    broad_prior = BROAD_PRIOR if broad_prior is None else broad_prior

    return (float(broad_prior["sigma"][1]), float(broad_prior["sigma"][2]))


def groups(hist, min_farms_per_region=MIN_FARMS_PER_REGION):
    """
    The farm-seasons, farms, and states a stage-1 fit will index.

    A state needs `min_farms_per_region` farms before it gets its own level; thinner states fall
    back to the global mean rather than fitting an offset to one or two farms. For a Season 25/26
    target that leaves AK, BC, CT, ME, NY, RI, and WA with levels and MA, MD, NS pooled globally.

    Args:
        hist (pd.DataFrame): historical events from `preprocess.history`.
        min_farms_per_region (int): farms a state needs for its own level.

    Returns:
        tuple: `(fs, farms, states)` -- one row per historical farm-season in the order the fit
            indexes by, and the sorted level labels.
    """
    fs = (hist.drop_duplicates("farm_season_id")
              [["farm_season_id", "Farm Name", "State", "Season"]]
              .reset_index(drop=True))

    farms = sorted(fs["Farm Name"].unique())
    n_farms = fs.groupby("State")["Farm Name"].nunique()
    states = sorted(n_farms[n_farms >= min_farms_per_region].index)

    return fs, farms, states


def fit_hierarchy(hist, config=None, mcmc=None, broad_prior=None, seed=0,
                  min_farms_per_region=MIN_FARMS_PER_REGION,
                  level_sd_fraction=LEVEL_SD_FRACTION):
    """
    Stage 1: one joint fit over all the farm-seasons in `hist`.

    Every historical farm-season gets its own curve, built as global mean + state offset + farm
    offset + that season's deviation, and the sizes of those offsets (`sd_state_*`, `sd_farm_*`,
    `sd_season_*`) are themselves estimated. Learning those sizes is the point: they tell a new
    target how much farms, states, and seasons actually differ.

    Non-centred throughout (`mu + sd * z`) because the centred form funnels badly when a level's sd
    can approach zero, which these genuinely can.

    `sigma` is one scalar shared by every historical farm-season's sample likelihood, not
    per-season. `sigma_h` is per-farm-season through the hierarchy.

    Args:
        hist (pd.DataFrame): historical events from `preprocess.history`. Must be strictly earlier
            than the target season or the backtest leaks.
        config (dict | None): `param -> levels`. Defaults to `CONFIG`. A season level is added to
            every parameter regardless of what this says.
        mcmc (dict | None): sampler settings overrides. Merged with `STAGE1_MCMC`.
        broad_prior (dict | None): the hyperprior source. Defaults to `BROAD_PRIOR`.
        seed (int): NUTS seed, exposed so a seed study can vary it.
        min_farms_per_region (int): passed to `groups`.
        level_sd_fraction (float): fraction of the broad prior's spread allowed between levels.

    Returns:
        tuple: `(idata, (fs, farms, states))`. The group tuple is needed to index a farm or state in
            stage 2.
    """
    config = CONFIG if config is None else config
    center, spread = hyperprior(broad_prior)
    level_sd = {p: level_sd_fraction * spread[p] for p in ORDER}
    fs, farms, states = groups(hist, min_farms_per_region)
    farm_idx = fs["Farm Name"].map({f: i for i, f in enumerate(farms)}).values
    state_idx = (fs["State"].map({s: i for i, s in enumerate(states)})
                 .fillna(len(states)).astype(int).values)
    obs = pd.concat([pre.season_obs(hist, f).assign(fs=i)
                     for i, f in enumerate(fs["farm_season_id"])])
    s, h = obs[obs.kind == "sample"], obs[obs.kind == "harvest"]
    m = {**STAGE1_MCMC, **(mcmc or {})}

    coords = {"farm_season": fs["farm_season_id"], "farm": farms, "state": states}

    with pm.Model(coords=coords):
        theta = {}
        for p in ORDER:
            val = pm.Normal(f"mu_{p}", center[p], spread[p])
            if "state" in config[p] and states:
                sd = pm.HalfNormal(f"sd_state_{p}", level_sd[p])
                z = pm.Normal(f"z_state_{p}", 0, 1, dims="state")
                # the trailing zero is the fallback level for thin states
                val = val + pt.concatenate([sd * z, pt.zeros(1)])[state_idx]
            if "farm" in config[p]:
                sd = pm.HalfNormal(f"sd_farm_{p}", level_sd[p])
                z = pm.Normal(f"z_farm_{p}", 0, 1, dims="farm")
                val = val + (sd * z)[farm_idx]
            sd = pm.HalfNormal(f"sd_season_{p}", level_sd[p])
            z = pm.Normal(f"z_season_{p}", 0, 1, dims="farm_season")
            theta[p] = pm.Deterministic(p, val + sd * z, dims="farm_season")

        A, k = pt.exp(theta["logA"]), pt.exp(theta["logk"])
        t0, c = theta["t0"], pt.exp(theta["logc"])
        sigma_h = pt.exp(theta["logsh"])
        sigma = pm.LogNormal("sigma", *sigma_prior(broad_prior))
        i, j = s["fs"].values, h["fs"].values
        pm.StudentT("y_s", nu=NU, sigma=sigma, observed=s["y"].values,
                    mu=logistic(s["day"].values, A[i], k[i], t0[i], m=pm.math))
        pm.StudentT("y_h", nu=NU, sigma=sigma_h[j], observed=h["y"].values,
                    mu=c[j] * logistic(h["day"].values, A[j], k[j], t0[j], m=pm.math))
        idata = pm.sample(draws=m["draws"], tune=m["tune"], chains=m["chains"],
                          cores=m["cores"], target_accept=m["target_accept"],
                          random_seed=seed, progressbar=False,
                          compute_convergence_checks=False)

    return idata, (fs, farms, states)


def run_stage1(events, season, cache_dir, run_prefix="H_kelp", mcmc=None,
               config=None, broad_prior=None, seed=0, refit=False):
    """
    Stage 1 for one target season, cached to disk.

    `preprocess.history` is what makes this honest: only farm-seasons from strictly earlier seasons.
    That is also why there is one fit per target year rather than one overall, since a shared fit
    would leak later seasons into earlier targets' priors. The fit depends only on the target year,
    so it is reused across every farm-season in that year.

    Only `mu_*`, `sd_*`, `z_state_*`, and `z_farm_*` draws are kept. The per-season `z`s are not
    needed, since a new season's deviation is redrawn rather than reused.

    Args:
        events (pd.DataFrame): the full events table. Filtered to history internally.
        season (str): the target season, e.g. "Season 25/26".
        cache_dir (str | Path): where the pickle lives.
        run_prefix (str): filename prefix, so runs with different settings do not overwrite.
        mcmc (dict | None): sampler settings. Appears in the filename.
        config (dict | None): passed to `fit_hierarchy`.
        broad_prior (dict | None): passed to `fit_hierarchy`.
        seed (int): NUTS seed.
        refit (bool): ignore an existing cache file and resample. Defaults to False.

    Returns:
        dict: `post` (flattened draws), `groups`, `summary`, and `divergences`. Also written to
            disk.
    """
    mcmc = mcmc or STAGE1_MCMC
    tag = f"d{mcmc['draws']}_t{mcmc['tune']}_c{mcmc['chains']}"
    path = Path(cache_dir) / f"stage1_{run_prefix}_{tag}_{season[-5:].replace('/', '')}.pkl"

    if path.exists() and not refit:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    idata, g = fit_hierarchy(pre.history(events, season), config=config,
                             mcmc=mcmc, broad_prior=broad_prior, seed=seed)
    post = {v: idata.posterior[v].values.reshape(-1, *idata.posterior[v].shape[2:])
            for v in idata.posterior.data_vars if v.startswith(_KEEP_PREFIXES)}
    hyper = [v for v in idata.posterior.data_vars if v.startswith(("mu_", "sd_"))]
    out = {"post": post, "groups": g,
           "summary": az.summary(idata, var_names=hyper + ["sigma"], kind="all"),
           "divergences": int(idata.sample_stats.diverging.values.sum())}

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as fh:
        pickle.dump(out, fh)

    return out


def stage1_health(stage1):
    """
    Convergence of each cached stage-1 fit. Check before trusting a prior.

    These variance components funnel: two levels explaining the same spread are only weakly
    distinguishable, and a level sd that can approach zero is a geometry NUTS handles badly even
    non-centred. It matters more than ordinary diagnostics because a poorly mixed `sd_season_*` sets
    the width of the prior it feeds, so bad mixing becomes a wrong range downstream.

    Season 25/26 is the known weak one at the default tune; raising tune to 2000-4000 resolves it.

    Args:
        stage1 (dict): `season -> run_stage1` output.

    Returns:
        pd.DataFrame: one row per season with the worst r_hat, lowest ess, divergences, and the
            three worst-mixed parameter names.
    """
    rows = []

    for season, fit in stage1.items():
        s = fit["summary"]
        bad = s[(s.r_hat > 1.01) | (s.ess_bulk < 400)]
        rows.append({"season": season, "history_farm_seasons": len(fit["groups"][0]),
                     "farms": len(fit["groups"][1]),
                     "states": ", ".join(fit["groups"][2]),
                     "max_rhat": float(s["r_hat"].max()),
                     "min_ess_bulk": float(s["ess_bulk"].min()),
                     "divergences": fit["divergences"],
                     "params_flagged": int(len(bad)),
                     "worst": ", ".join(bad.sort_values("r_hat", ascending=False)
                                        .index[:3])})

    return pd.DataFrame(rows).set_index("season")


def new_season_draws(stage1_fit, farm, state, config=None, seed=0):
    """
    Stage 2a: stage 1's posterior, composed into a new season at this farm.

    Draws are global mean + state offset + farm offset + a fresh season deviation. The season term
    is drawn from its estimated sd rather than taken from any fitted season, because the target's
    own deviation is unknown and pretending otherwise is how a backtest leaks. It is also the
    largest of the four terms; for `A` it contributes about 4x what the state offset does.

    A farm with no history gets a fresh draw for its farm effect too, so it is treated as a draw
    from the farm population rather than as average. A state below the `groups` threshold
    contributes no offset at all.

    The target's own data is never involved.

    Args:
        stage1_fit (dict): one `run_stage1` output, for the target's season.
        farm (str): the target's farm name. Need not appear in the history.
        state (str): the target's state. Need not be a pooled state.
        config (dict | None): `param -> levels`. Defaults to `CONFIG`.
        seed (int): seeds the fresh season and absent-farm draws.

    Returns:
        pd.DataFrame: one column per name in `ORDER`, one row per stage-1 draw.
    """
    config = CONFIG if config is None else config
    post, (fs, farms, states) = stage1_fit["post"], stage1_fit["groups"]
    rng = np.random.default_rng(seed)
    n = len(post["mu_logA"])
    out = {}

    for p in ORDER:
        v = post[f"mu_{p}"].copy()
        if "state" in config[p] and state in states:
            v = v + post[f"sd_state_{p}"] * post[f"z_state_{p}"][:, states.index(state)]
        if "farm" in config[p]:
            z = (post[f"z_farm_{p}"][:, farms.index(farm)] if farm in farms
                 else rng.standard_normal(n))
            v = v + post[f"sd_farm_{p}"] * z
        v = v + post[f"sd_season_{p}"] * rng.standard_normal(n)
        out[p] = v

    return pd.DataFrame(out)


def mvn_prior(draws, broad_prior=None):
    """
    Stage 2b: those draws as a multivariate normal PyMC can use as a prior.

    A Gaussian approximation, because stage 1 yields draws while PyMC needs a closed-form logp.
    Fitting it on the log scale means a lognormal in natural units, which matches these parameters'
    skew. It would misfit a genuinely bimodal or heavy-tailed predictive, worth checking against
    `new_season_draws` if a target looks strange.

    The full covariance is kept, not just the marginals, so correlations learned in stage 1 (notably
    `logA` with `t0`) carry into the target's prior. `sigma` is appended from the broad prior.

    Args:
        draws (pd.DataFrame): `new_season_draws` output.
        broad_prior (dict | None): source of the `sigma` prior. Defaults to `BROAD_PRIOR`.

    Returns:
        tuple: `(add_prior, mean, cov)`. `add_prior(dims=None)` must be called inside a `pm.Model`.
            `mean` and `cov` are what the cache fingerprint is built from.
    """
    mean = draws[ORDER].mean().values
    cov = np.cov(draws[ORDER].values.T)

    def add_prior(dims=None):
        """
        Add the fitted multivariate-normal prior to the open PyMC model.

        Args:
            dims: accepted for signature compatibility; unused, since one target is fitted at a
                time.

        Returns:
            dict: `param -> RV`, with the log-scale entries exponentiated back to natural units.
        """
        th = pm.MvNormal("theta", mu=mean, cov=cov)

        return {"A": pm.Deterministic("A", pt.exp(th[0])),
                "k": pm.Deterministic("k", pt.exp(th[1])),
                "t0": pm.Deterministic("t0", th[2]),
                "c": pm.Deterministic("c", pt.exp(th[3])),
                "sigma_h": pm.Deterministic("sigma_h", pt.exp(th[4])),
                "sigma": pm.LogNormal("sigma", *sigma_prior(broad_prior))}

    return add_prior, mean, cov


def hierarchical_prior_for(events, stage1, run_name="hierarchy", config=None,
                           broad_prior=None, seed=0):
    """
    Build `prior_for(fsid) -> model.Prior`, the shape every report function takes.

    The hierarchy's entry point: one callable that hands any farm-season its own prior, so
    `SeasonModel` and the evaluation functions need no special-casing.

    Falls back to the unpooled broad prior when the target's season has no stage-1 fit, i.e. the
    earliest year. The trigger is the season, not the farm: a target whose farm is absent from the
    history still gets a hierarchical prior, with a fresh draw for its farm effect.

    Each farm-season gets its own fingerprint from its mean and covariance, so hierarchical and
    unpooled posteriors for the same farm-season cannot collide in one cache file. The fingerprint
    comes from the fitted definition rather than samples of it, so it does not depend on `seed`.

    Args:
        events (pd.DataFrame): the full events table, for farm, season, and state metadata.
        stage1 (dict): `season -> run_stage1` output. A missing season triggers the fallback.
        run_name (str): label that appears in each prior's `source` string and so in reports.
        config (dict | None): passed to `new_season_draws`.
        broad_prior (dict | None): hyperprior and fallback source. Defaults to `BROAD_PRIOR`.
        seed (int): passed to `new_season_draws`.

    Returns:
        callable: `get(fsid) -> model.Prior`, memoised per farm-season.
    """
    meta_by_fsid = events.drop_duplicates("farm_season_id").set_index("farm_season_id")
    memo = {}

    prior_def = BROAD_PRIOR if broad_prior is None else broad_prior

    def build_for(fsid):
        """
        Compose one farm-season's prior builder, source label, and fingerprint.

        Args:
            fsid (str): farm-season id.

        Returns:
            tuple: `(add_prior, source, fingerprint)`. The unpooled broad prior when the season has
                no stage-1 fit.
        """
        m = meta_by_fsid.loc[fsid]

        if m["Season"] not in stage1:
            # the earliest season year has no history to learn from, so it
            # falls back to the unpooled broad prior rather than special-casing
            def broad(dims=None):
                """
                The unpooled broad prior, as a stage-2 shaped builder.

                Args:
                    dims: passed through to `model.make_prior_for_season`.

                Returns:
                    dict: `param -> RV`. Must be called inside a `pm.Model`.
                """
                return make_prior_for_season(None, None, None, dims=dims,
                                             broad_prior=prior_def)
            return broad, "population (domain ranges)", broad_prior_fingerprint(prior_def)

        draws = new_season_draws(stage1[m["Season"]], m["Farm Name"], m["State"],
                                 config=config, seed=seed)
        add_prior, mean, cov = mvn_prior(draws, prior_def)
        # fingerprint the FITTED definition, not samples from it
        fp = broad_prior_fingerprint({p: ("normal", mean[i], float(np.sqrt(cov[i, i])))
                                      for i, p in enumerate(ORDER)})

        return add_prior, f"{run_name} ({m['Farm Name']}, {m['State']})", fp

    def get(fsid):
        """
        Memoised `Prior` for one farm-season.

        Args:
            fsid (str): farm-season id.

        Returns:
            model.Prior: the builder plus its metadata.
        """
        if fsid not in memo:
            m = meta_by_fsid.loc[fsid]
            build, source, fp = build_for(fsid)
            memo[fsid] = Prior(build, {"method": "hierarchy", "source": source,
                                       "fingerprint": fp, "farm": m["Farm Name"],
                                       "season": m["Season"], "state": m["State"]})

        return memo[fsid]

    return get


def prior_for(events, stage1, run_name="hierarchy", config=None,
              broad_prior=None, seed=0):
    """
    Deprecated alias for `hierarchical_prior_for`.

    Renamed because "prior_for" collided with the unpooled builder of the same name once both models
    were being run side by side.

    Args:
        events (pd.DataFrame): see `hierarchical_prior_for`.
        stage1 (dict): see `hierarchical_prior_for`.
        run_name (str): see `hierarchical_prior_for`.
        config (dict | None): see `hierarchical_prior_for`.
        broad_prior (dict | None): see `hierarchical_prior_for`.
        seed (int): see `hierarchical_prior_for`.

    Returns:
        callable: whatever `hierarchical_prior_for` returns.
    """
    return hierarchical_prior_for(events, stage1, run_name=run_name,
                                  config=config, broad_prior=broad_prior,
                                  seed=seed)
