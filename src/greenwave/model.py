"""The model: growth curve, priors, fitting, prediction."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .preprocess import (
    HARVEST_EVENT,
    IDCOL,
    SAMPLE_EVENT,
    TCOL,
    data_fingerprint,
    season_events,
    split,
)

CURVE = "logistic"
C_PARAM = "c"
#: Student-t degrees of freedom for both observation channels. Heavy tails:
#: one bad reading is tolerated rather than allowed to bend the curve.
NU = 4


def logistic(t, A, k, t0, m=np):
    """
    Symmetric sigmoid growth curve, the model's `f(t)`.

    `A / (1 + exp(-k (t - t0)))`. Symmetric about `t0`.

    Args:
        t (float | np.ndarray): day of season.
        A (float | np.ndarray): plateau in lbs/ft.
        k (float | np.ndarray): growth rate per day.
        t0 (float | np.ndarray): inflection day, where growth is fastest.
        m (module): `np` to evaluate, `pytensor.tensor` to build a PyMC graph. Defaults to `np`.

    Returns:
        float | np.ndarray: yield in lbs/ft.
    """
    return A / (1.0 + m.exp(-k * (t - t0)))


def gompertz(t, A, k, t0, m=np):
    """
    Asymmetric sigmoid: fast start, slow finish, long right taper.

    `A * exp(-exp(-k (t - t0)))`. Same parameter names as `logistic`.

    Args:
        t (float | np.ndarray): day of season.
        A (float | np.ndarray): plateau in lbs/ft.
        k (float | np.ndarray): growth rate per day.
        t0 (float | np.ndarray): inflection day.
        m (module): `np` or `pytensor.tensor`. Defaults to `np`.

    Returns:
        float | np.ndarray: yield in lbs/ft.
    """
    return A * m.exp(-m.exp(-k * (t - t0)))


# Current curve options
CURVES = {"logistic": logistic, "gompertz": gompertz}


BROAD_PRIOR = {
    # 95% ~ 0.83 - 19 lbs/ft plateau
    "A": ("lognormal", np.log(4.0), 0.80),
    # 95% ~ 0.011 - 0.22 /day; k trades off against t0 along a ridge
    "k": ("lognormal", np.log(0.05), 0.75),
    # inflection late Dec - May; untruncated (P(t0<0) = 0.04%)
    "t0": ("normal", 150.0, 45.0),
    # harvest/sample scale, 95% ~ 0.33 - 1.3, deliberately asymmetric about 1
    "c": ("lognormal", np.log(0.65), 0.35),
    # sample noise, 95% ~ 0.10 - 2.4 lbs/ft against readings of ~0.3 - 2
    "sigma": ("lognormal", np.log(0.5), 0.80),
    # same spec as `sigma`: too few farm-seasons have enough harvests to
    # estimate a separate one, so it is tied rather than pretended
    "sigma_h": ("lognormal", np.log(0.5), 0.80),
}


# parameters defined on the log scale (family "lognormal")
LOG_SCALE_PARAMS = tuple(p for p, d in BROAD_PRIOR.items() if d[0] == "lognormal")


def build_prior_rvs(broad_prior=None, dims=None):
    """
    Turn a prior specification into live PyMC random variables.

    Must be called inside a `pm.Model()`, since the RVs register with whichever model is on the
    context stack. That is why priors are passed around as callables rather than objects.

    `posterior_mcmc` opens one model per farm-season, so every parameter here, `sigma` and `sigma_h`
    included, is a scalar fitted to that farm-season alone.

    Args:
        broad_prior (dict | None): `param -> (family, a, b)`, family "lognormal" or "normal".
            Defaults to `BROAD_PRIOR`.
        dims (str | tuple | None): PyMC dims for the curve parameters, for a fit vectorised over
            farm-seasons. No caller passes it, so it is a seam rather than a live path.

    Returns:
        dict: `param -> pm RV`.
    """
    broad_prior = BROAD_PRIOR if broad_prior is None else broad_prior
    out = {}

    for name, (family, a, b) in broad_prior.items():
        # sigma and sigma_h are scalars even when the curve parameters are
        # vectorised over farm-seasons
        d = None if name in ("sigma", "sigma_h") else dims
        if family == "lognormal":
            out[name] = pm.LogNormal(name, a, b, dims=d)
        elif family == "normal":
            out[name] = pm.Normal(name, a, b, dims=d)
        else:
            raise ValueError(f"unknown prior family {family!r} for {name}")

    return out


def broad_prior_fingerprint(broad_prior):
    """
    Hash of a prior's definition.

    To determine if a prior has been changed and invalidates cached results.

    Args:
        broad_prior (dict): `param -> (family, a, b)`.

    Returns:
        str: first 8 hex characters of the SHA-1 of the sorted spec.
    """
    payload = repr(
        sorted(
            (k, v[0], round(float(v[1]), 12), round(float(v[2]), 12))
            for k, v in broad_prior.items()
        )
    )

    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:8]


def harvest_scale(draws, c_param=None):
    """
    Multiplicative harvest scale per posterior or prior draw.

    Args:
        draws (dict): a posterior or prior draw dict containing "c".
        c_param (str | None): "c", "harvest_factor", or None. Anything else raises, to catch a call
            site expecting a parameterisation that was never implemented. Only use "c", leftover
            from trying different functions with a harvest calibration factor.

    Returns:
        np.ndarray: `c` per draw.
    """
    if c_param not in (None, "c", "harvest_factor"):
        raise ValueError(f"unknown harvest-scale parameterization {c_param!r}")

    return np.asarray(draws["c"], float)


def posterior_mcmc(
    define_priors,
    curve_name,
    t,
    y,
    th=None,
    yh=None,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.9,
    rng=0,
    progressbar=False,
    c_param=None,
):
    """
    NUTS posterior for one farm-season.

    Builds the two-channel likelihood and samples it: samples against `f(t)` with `sigma`, harvests
    against `c * f(t)` with `sigma_h`. Student-t with `NU` = 4 rather than normal, so one odd
    reading is tolerated instead of bending the curve.

    Args:
        define_priors (callable): returns `{name: pm RV}` for A, k, t0, c, sigma, sigma_h. Called
            inside this model so the RVs register here. Any pm distribution works.
        curve_name (str): key into `CURVES`.
        t (array-like): sample days.
        y (array-like): sample yields in lbs/ft.
        th (array-like | None): harvest days. None fits samples only.
        yh (array-like | None): harvest yields in lbs/ft.
        draws (int): posterior draws per chain, after tuning.
        tune (int): warm-up draws, discarded. NUTS adapts its step size and mass matrix here.
        chains (int): independent chains. More than one is what makes `r_hat` meaningful.
        target_accept (float): acceptance rate NUTS adapts toward. Higher means smaller steps.
        rng (int): NUTS seed, and the seed for the no-harvest prior fill.
        progressbar (bool): show PyMC's progress bar.
        c_param (str | None): validated early, so a bad name fails before sampling.

    Returns:
        dict: flattened draws per parameter, plus `rhat_max`, `divergences`, `ess`, and
            `method="mcmc"`.
    """
    harvest_scale({"c": np.array([1.0])}, c_param)  # validate the name early, leftover
    curve_fn = CURVES[curve_name]
    t, y = np.asarray(t, float), np.asarray(y, float)
    has_h = th is not None and yh is not None and len(np.atleast_1d(th)) > 0

    if has_h:
        th, yh = np.asarray(th, float), np.asarray(yh, float)

    scale_name = "c"

    with pm.Model():
        pri = define_priors()
        A, k, t0, sigma = pri["A"], pri["k"], pri["t0"], pri["sigma"]
        # calculate the likelihood of the data given the parameters
        if len(t):
            pm.StudentT("obs_s", nu=NU, mu=curve_fn(t, A, k, t0, m=pt), sigma=sigma, observed=y)
        if has_h:
            scale = pri[scale_name]
            pm.StudentT(
                "obs_h",
                nu=NU,
                mu=scale * curve_fn(th, A, k, t0, m=pt),
                sigma=pri["sigma_h"],
                observed=yh,
            )
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=rng,
            progressbar=progressbar,
            compute_convergence_checks=False,
        )

    # get the posterior distributions per model parameter
    post = {}

    for p in ["A", "k", "t0", "sigma", "sigma_h", scale_name]:
        if p in idata.posterior:
            post[p] = idata.posterior[p].to_numpy().reshape(-1)

    n = len(post["A"])

    if scale_name not in post or "sigma_h" not in post:
        # no harvests observed, stay at the prior
        pri_draws = sample_prior(define_priors, n=n, rng=rng)
        post.setdefault(scale_name, pri_draws.get(scale_name, np.full(n, np.nan)))
        post.setdefault("sigma_h", pri_draws.get("sigma_h", np.full(n, np.nan)))

    summ = az.summary(
        idata,
        var_names=[v for v in ["A", "k", "t0", scale_name] if v in idata.posterior],
        kind="diagnostics",
    )
    post["rhat_max"] = float(summ["r_hat"].max())
    post["divergences"] = int(idata.sample_stats.diverging.to_numpy().sum())
    post["ess"] = float(summ["ess_bulk"].min())
    post["method"] = "mcmc"

    return post


def make_prior_for_season(
    cloud=None,
    farm=None,
    before_season=None,
    method="broad",
    dims=None,
    verbose=False,
    broad_prior=None,
):
    """
    The prior for one target farm-season, as PyMC random variables.

    Must be called inside a `pm.Model()`; `posterior_mcmc` does that. Pass it as a callable rather
    than calling it yourself. Only the "broad" method exists here, the hand-set ranges in
    `BROAD_PRIOR`. They are never fitted to this dataset, so there is no leakage to reason about.
    Partial pooling lives in `hierarchy`.

    Args:
        cloud (pd.DataFrame | None): unused. Kept as the seam for partial pooling.
        farm (str | None): unused except in the `verbose` line.
        before_season (str | None): unused except in the `verbose` line.
        method (str): only "broad" is implemented.
        dims (str | tuple | None): passed to `build_prior_rvs`.
        verbose (bool): print which prior was chosen.
        broad_prior (dict | None): override the spec. Defaults to `BROAD_PRIOR`.

    Returns:
        dict: `param -> pm RV`.
    """
    if method != "broad":
        raise ValueError(
            f"unknown prior method {method!r} (only 'broad'; "
            f"partial pooling is not implemented yet)"
        )

    if verbose:
        print(f"  {farm} {before_season}: broad prior (no history used)")

    return build_prior_rvs(broad_prior, dims=dims)


def sample_prior(define_priors, n=4000, rng=None):
    """
    Draw `n` samples from every prior, for the fan plots and no-data panels.

    Args:
        define_priors (callable): returns `{name: pm RV}`.
        n (int): draws per parameter. Defaults to 4000, matching a 1000x4 posterior.
        rng (int | None): seed. None means 0, deliberately deterministic since these draws feed
            reported ranges.

    Returns:
        dict: `param -> np.ndarray` of length `n`.
    """
    with pm.Model():
        pri = define_priors()

    seed = 0 if rng is None else rng

    if not isinstance(seed, int | np.integer):
        seed = int(np.random.default_rng(seed).integers(2**31))

    names = list(pri)
    drawn = pm.draw([pri[name] for name in names], n, random_seed=seed)

    return {name: np.asarray(d).reshape(-1) for name, d in zip(names, drawn)}


# sampler configuration defaults
SAMPLER_DEFAULTS = {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.9, "seed": 0}

# convergence thresholds the diagnostics report against
RHAT_OK = 1.01
RHAT_LOOSE = 1.05
ESS_MIN = 400


#: posterior array to cache
POSTERIOR_PARAMS_TO_KEEP = ("A", "k", "t0", "c", "sigma", "sigma_h")
# diagnostics carried alongside draws from the posterior
DIAGNOSTICS_TO_KEEP = ("method", "rhat_max", "divergences", "ess")
# positions in a `PosteriorCache.key` tuple
_KEY_PRIOR_SOURCE, _KEY_PRIOR_FINGERPRINT = 5, 6


class Prior:
    """
    A deferred prior for one farm-season, plus where it came from.

    The prior builders return live PyMC random variables, so they must run inside the model that
    uses them. This keeps that call deferred and carries the provenance the tables report.
    """

    def __init__(self, build, meta):
        """
        Wrap a prior builder with its provenance.

        Args:
            build (callable): `build(dims=None) -> {name: pm RV}`, called inside a `pm.Model()`.
            meta (dict): provenance. Must carry a truthy "fingerprint"; `source` and `method` are
                read by the cache key and the reports.
        """
        self._build, self._meta = build, dict(meta)

        if not self._meta.get("fingerprint"):
            raise ValueError(
                "Prior needs an explicit meta['fingerprint'] -- use "
                "model.broad_prior_fingerprint() on the prior's definition. "
                "A fingerprint derived from samples is not stable."
            )

    def __call__(self, dims=None):
        """
        Build the prior's random variables in the open PyMC model.

        Args:
            dims (str | tuple | None): PyMC dims, for builders that accept them.

        Returns:
            dict: `param -> pm RV`.
        """
        try:
            return self._build(dims)
        except TypeError:
            return self._build()  # builders that take no arguments

    def __getitem__(self, key):
        """
        Mapping access, so a `Prior` can stand in where a dict was expected.

        Only `prior["_meta"]` is supported, so the cache and report functions can read provenance
        off either shape.

        Args:
            key (str): must be "_meta".

        Returns:
            dict: the metadata.
        """
        if key != "_meta":
            raise KeyError(key)

        return self._meta

    @property
    def meta(self):
        """
        The provenance dict.

        Returns:
            dict: `method`, `source`, `fingerprint`, and whatever else the factory attached.
        """
        return self._meta


def unpooled_prior_for(broad_prior=None, method="broad"):
    """
    Build `prior_for(fsid) -> Prior` where every farm-season gets the same prior.

    Each farm-season is fitted independently and keeps its own parameters; the only thing shared is
    this prior. Nothing is pooled in the hierarchical sense. Contrast
    `hierarchy.hierarchical_prior_for`.

    Takes no events table on purpose. This prior cannot see data, and an `events` argument would
    suggest otherwise.

    Args:
        broad_prior (dict | None): override the spec from a notebook. The fingerprint follows
            automatically, so an edited prior cannot collide with the original in a cache.
        method (str): passed to `make_prior_for_season`; only "broad" works.

    Returns:
        callable: `prior_for(fsid=None) -> Prior`. The `fsid` is ignored, and is in the signature so
            unpooled and hierarchical factories are interchangeable at every call site.
    """
    prior_def = BROAD_PRIOR if broad_prior is None else broad_prior
    fingerprint = broad_prior_fingerprint(prior_def)
    source = "population (domain ranges)" if method == "broad" else f"{method} prior"

    def build(dims=None):
        """
        Build the broad prior's RVs in the open PyMC model.

        Args:
            dims (str | tuple | None): passed to `make_prior_for_season`.

        Returns:
            dict: `param -> pm RV`.
        """
        return make_prior_for_season(
            None, None, None, method=method, dims=dims, broad_prior=prior_def
        )

    def prior_for(fsid=None):
        """
        The broad `Prior`, identical for every farm-season.

        Args:
            fsid (str | None): ignored; accepted to match the hierarchical factory's signature.

        Returns:
            Prior: the deferred builder plus its metadata.
        """
        return Prior(
            build,
            {"method": method, "source": source, "fingerprint": fingerprint, "c_param": C_PARAM},
        )

    return prior_for


def prior_summary(prior, n=20000, rng=0, params=None):
    """
    Median and 95% range per parameter, for any prior.

    Works on the unpooled broad prior and a hierarchical one alike.

    Args:
        prior (callable | Prior): anything `sample_prior` accepts.
        n (int): draws. Defaults to 20000, more than a posterior needs because these are cheap.
        rng (int): seed. Defaults to 0.
        params (list[str] | None): which parameters, in order. Defaults to all six present.

    Returns:
        pd.DataFrame: indexed by `param`, with `median`, `lo95`, `hi95`, `width`.
    """
    draws = sample_prior(prior, n=n, rng=rng)
    params = params or [p for p in ("A", "k", "t0", "c", "sigma", "sigma_h") if p in draws]
    rows = []

    for p in params:
        lo, med, hi = np.percentile(draws[p], [2.5, 50, 97.5])
        rows.append({"param": p, "median": med, "lo95": lo, "hi95": hi, "width": hi - lo})

    return pd.DataFrame(rows).set_index("param")


def compare_priors(priors, n=20000, rng=0, params=None):
    """
    Several priors side by side: median and 95% per parameter.

    This is how you see that a hierarchical prior tightens some parameters and widens others, which
    a single prior's summary cannot tell you.

    Args:
        priors (dict): `label -> prior builder`.
        n (int): draws per prior. Defaults to 20000.
        rng (int): seed, shared across priors so differences are not noise.
        params (list[str] | None): which parameters.

    Returns:
        pd.DataFrame: indexed by `param`, with a two-level `(label, statistic)` column index.
    """
    out = {}

    for label, prior in priors.items():
        s = prior_summary(prior, n=n, rng=rng, params=params)
        for col in ("median", "lo95", "hi95"):
            out[(label, col)] = s[col]

    t = pd.DataFrame(out)
    t.columns = pd.MultiIndex.from_tuples(t.columns)

    return t


def fit_posterior(prior, seen, sampler=None, curve=CURVE, c_param=C_PARAM):
    """
    Prior plus data to posterior draws. No cache involved.

    Args:
        prior (callable | Prior): the deferred prior builder.
        seen (list[tuple]): observations to condition on, from `preprocess.season_events`.
        sampler (dict | None): sampler settings overrides. Merged with `SAMPLER_DEFAULTS`.
        curve (str): key into `CURVES`.
        c_param (str): harvest-scale parameterisation.

    Returns:
        dict: `posterior_mcmc`'s output, or prior draws with `method="prior"` when `seen` is empty.
    """
    s = {**SAMPLER_DEFAULTS, **(sampler or {})}
    ts, ys, th, yh = split(seen)

    if len(ts) == 0 and len(th) == 0:
        # seeded: an unseeded prior draw makes the no-data row irreproducible
        post = sample_prior(prior, n=4000, rng=s["seed"])
        post["method"] = "prior"
        return post

    return posterior_mcmc(
        prior,
        curve,
        ts,
        ys,
        th if len(th) else None,
        yh if len(yh) else None,
        c_param=c_param,
        draws=s["draws"],
        tune=s["tune"],
        chains=s["chains"],
        target_accept=s["target_accept"],
        rng=s["seed"],
    )


class PosteriorCache:
    """
    Posteriors memorised per farm-season and prefix, persistable to disk.

    The key pins everything that determines the draws, so changing any of them gives a miss rather
    than a wrong answer.
    """

    def __init__(self, sampler=None, curve=CURVE, c_param=C_PARAM, mcmc=None):
        """
        An empty posterior store bound to one sampler configuration.

        Args:
            sampler (dict | None): sampler settings. Merged with `SAMPLER_DEFAULTS`. Part of every
                key, so a cache holds one configuration unless a call overrides it per-`get`.
            curve (str): key into `CURVES`.
            c_param (str): harvest-scale parameterisation.
            mcmc (dict | None): the old name for `sampler`, still accepted.
        """
        self.store: dict = {}

        self.sampler = {**SAMPLER_DEFAULTS, **(sampler or mcmc or {})}
        self.curve, self.c_param = curve, c_param

        self.hits = 0
        self.misses = 0
        self.last_loaded: list = []
        self.last_saved = None

    @property
    def mcmc(self):
        """
        Back-compatible alias for `sampler`. Used to use mcmc. Now generalized to sampler.

        Returns:
            dict: the sampler configuration.
        """
        return self.sampler

    def key(self, fsid, n_seen, seen, prior, sampler=None):
        """
        What uniquely determines these posterior draws.

        Args:
            fsid (str): farm-season id.
            n_seen (int): how many observations were conditioned on.
            seen (list[tuple]): those observations, hashed into the key.
            prior (Prior | dict): read for its `source` and `fingerprint`.
            sampler (dict | None): per-call overrides merged over the cache's own.

        Returns:
            tuple: the hashable key, length 8. `KEY_FINGERPRINT_INDEX` is where the prior
                fingerprint sits.
        """
        s = {**self.sampler, **(sampler or {})}

        return (
            fsid,
            n_seen,
            data_fingerprint(seen),
            self.curve,
            POSTERIOR_PARAMS_TO_KEEP,
            prior["_meta"]["source"],
            prior["_meta"].get("fingerprint"),
            tuple(sorted(s.items())),
        )

    def get(self, fsid, events, n_seen, prior, sampler=None, mcmc=None):
        """
        Posterior for `fsid` given its first `n_seen` observations.

        Checks if posterior has already been sampled. Counts misses and hits to not repeat posterior
        fits.

        Args:
            fsid (str): farm-season id.
            events (list[tuple]): the farm-season's full observation list, sliced to `n_seen` here.
            n_seen (int): how many of them to condition on.
            prior (Prior): the prior for this farm-season.
            sampler (dict | None): per-call sampler overrides.
            mcmc (dict | None): the old name for `sampler`.

        Returns:
            dict: draws per kept parameter plus whichever diagnostics exist
        """
        sampler = sampler or mcmc
        seen = events[:n_seen]
        k = self.key(fsid, n_seen, seen, prior, sampler)

        if k in self.store:
            self.hits += 1
        else:
            self.misses += 1
            post = fit_posterior(
                prior, seen, {**self.sampler, **(sampler or {})}, self.curve, self.c_param
            )
            self.store[k] = {p: post[p] for p in POSTERIOR_PARAMS_TO_KEEP if p in post} | {
                p: post[p] for p in DIAGNOSTICS_TO_KEEP if p in post
            }

        return self.store[k]

    def counters(self):
        """
        Cumulative hits and fits so far. Snapshot before a loop, diff after.

        Returns:
            tuple[int, int]: `(hits, fitted)` since construction.
        """
        return self.hits, self.misses

    def report_since(self, before, what=""):
        """
        Print what happened since a `counters()` snapshot.

        Prints nothing when neither counter moved, so a fully cached call stays quiet.

        Args:
            before (tuple[int, int]): an earlier `counters()` result.
            what (str): label for the printout, e.g. "harvest_predictions".

        Returns:
            tuple[int, int]: `(hits, fitted)` over that interval.
        """
        hits, fitted = self.hits - before[0], self.misses - before[1]

        if hits or fitted:
            print(
                f"  cache{' for ' + what if what else ''}: "
                f"{hits} hit{'' if hits == 1 else 's'}, {fitted} fitted"
                f"  ({len(self.store)} stored)"
            )

        return hits, fitted

    def summary(self):
        """
        Where this cache came from, what is in it, and what it has done.

        Returns:
            dict: entry count, hits, fits, every file read into it, where it was saved, and the
                sampler tag.
        """
        return {
            "entries": len(self.store),
            "hits": self.hits,
            "fitted": self.misses,
            "loaded_from": list(self.last_loaded),
            "saved_to": self.last_saved,
            "sampler": sampler_tag(self.sampler),
        }

    def posterior(self, prior, seen, fsid=None, sampler=None):
        """
        Prior plus data to posterior, memorised.

        Args:
            prior (Prior): the prior for this farm-season.
            seen (list[tuple]): the observations to condition on.
            fsid (str | None): farm-season id, for the key. Defaults to "-".
            sampler (dict | None): per-call sampler overrides.

        Returns:
            dict: the posterior, as from `get`.
        """
        return self.get(fsid or "-", seen, len(seen), prior, sampler)

    def save(self, name, directory, overwrite=False):
        """
        Write the store to `<directory>/<name>.pkl`.

        Refuses to write if the target already holds entries this store does not, since that would
        drop another run's posteriors. Appending to a file from a different run is how a cache
        becomes something you cannot reason about, so it has to be asked for explicitly.

        Arrays are downcast to float32 on the way out, roughly halving the file; `load` restores
        them.

        Args:
            name (str): filename without the extension. Use `cache_name` to build one.
            directory (str | Path): created if absent.
            overwrite (bool): skip the safety check and write anyway. Defaults to False.

        Returns:
            Path: the file written, also printed with its size.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.pkl"

        if path.exists() and not overwrite:
            with path.open("rb") as fh:
                existing = set(pickle.load(fh))  # noqa: S301 -- the project's own cache file
            lost = existing - set(self.store)
            if lost:
                raise FileExistsError(
                    f"{path.name} holds {len(lost)} posteriors this cache does "
                    f"not have -- writing would drop them. Load it first "
                    f"(cache.load(...)), pick a different name "
                    f"(model.cache_name(...)), or pass overwrite=True."
                )

        slim = {
            k: {
                kk: (vv.astype(np.float32) if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()
            }
            for k, v in self.store.items()
        }

        with path.open("wb") as fh:
            pickle.dump(slim, fh, protocol=4)

        self.last_saved = str(path.resolve())
        print(
            f"SAVED {len(slim)} posteriors -> {path.resolve()} "
            f"({path.stat().st_size / 1e6:,.1f} MB)"
        )

        return path

    def load(self, name, directory, merge=True, migrate=None):
        """
        Read a cache file into the store.

        Prints what was loaded and how many distinct prior fingerprints it holds. More than one
        means the file mixes priors, which is worth knowing before trusting a table built from it.

        Args:
            name (str): filename without the extension.
            directory (str | Path): where to look. A missing file prints and returns 0 rather than
                raising, so a first run needs no special-casing.
            merge (bool): keep existing entries. Defaults to True; False clears first.
            migrate (callable | None): `old_key -> new_key or None`, a hook for reading a file
                written before a key change. Keys it returns None for are dropped rather than
                guessed at, so a stale entry can never be served as a current one. No cache in this
                repo needs it.

        Returns:
            int: how many entries were loaded.
        """
        path = Path(directory) / f"{name}.pkl"

        if not path.exists():
            print(f"no cache at {path.resolve()}")
            return 0

        with path.open("rb") as fh:
            loaded = pickle.load(fh)  # noqa: S301 -- the project's own cache file

        if migrate is not None:
            out, dropped = {}, 0
            for k, v in loaded.items():
                nk = migrate(k)
                if nk is None:
                    dropped += 1
                else:
                    out[nk] = v
            print(
                f"migrated {len(out)} keys from {path.name}"
                + (f", dropped {dropped} that no longer match the data" if dropped else "")
            )
            loaded = out

        loaded = {
            k: {
                kk: (vv.astype(float) if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()
            }
            for k, v in loaded.items()
        }

        if not merge:
            self.store.clear()

        self.store.update(loaded)
        fps = {
            k[KEY_FINGERPRINT_INDEX]
            for k in loaded
            if isinstance(k, tuple) and len(k) > KEY_FINGERPRINT_INDEX
        }
        self.last_loaded.append(str(path.resolve()))
        print(
            f"LOADED {len(loaded)} posteriors from {path.resolve()} "
            f"({len(fps)} distinct prior fingerprint"
            f"{'s' if len(fps) != 1 else ''}: {', '.join(sorted(map(str, fps)))})"
        )

        return len(loaded)

    def clear(self):
        """
        Drop every stored posterior.

        Leaves the hit and miss counters alone, since they describe this session's work rather than
        the store's contents.
        """
        self.store.clear()

    def __len__(self):
        """
        How many posteriors are stored.

        Returns:
            int: entry count.
        """
        return len(self.store)


# Claude's code to hash a posterior run!
KEY_FINGERPRINT_INDEX = 6  # (fsid, n_seen, data_fp, curve, KEEP, source, FP, sampler)


def q95(x):
    """
    Median and the 2.5 to 97.5 percentile range over the finite draws.

    Non-finite draws are dropped rather than propagated, so a season with no harvests, where `c` and
    `sigma_h` may be NaN, yields blanks instead of poisoning the whole range.

    Args:
        x (array-like): draws of any shape; flattened.

    Returns:
        tuple[float, float, float]: `(median, lo, hi)`, all NaN if nothing is finite.
    """
    v = np.asarray(x, float).reshape(-1)
    v = v[np.isfinite(v)]

    if not len(v):
        return np.nan, np.nan, np.nan

    med, lo, hi = np.quantile(v, [0.5, 0.025, 0.975])

    return med, lo, hi


def noise_rng(*key):
    """
    A reproducible generator, keyed by whatever you pass.

    The predictive ranges add simulated `sigma_h` noise, so they need randomness, but the same
    randomness every time or a table's numbers would shift between runs for no reason. Hashing a
    descriptive key gives independent streams per farm-season, day, and purpose with no counter to
    manage.

    Args:
        *key: any values; stringified and joined into the hash.

    Returns:
        np.random.Generator: seeded from the key's SHA-256.
    """
    h = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()

    return np.random.default_rng(int(h[:12], 16))


def harvest_intervals(post, day, rng_key, c_param=C_PARAM, curve=CURVE, nu=None):
    """
    The two 95% ranges for harvest lbs/ft on `day`.

    The distinction the reports turn on. The yield range is `c * f(day)` per draw, where the farm's
    true average harvest density that day sits. The reading range adds `sigma_h * t(nu)` noise and
    clips at 0: where a single cut can land, and the only fair range to score one harvest against.
    The clip is why early-season lower bounds read exactly 0.00 rather than something small.

    Args:
        post (dict): a posterior with A, k, t0, c, sigma_h.
        day (float): day of season to evaluate at.
        rng_key (tuple): key for `noise_rng`, so the reading range is reproducible. Include the
            farm-season, day, and prefix.
        c_param (str): harvest-scale parameterisation.
        curve (str): key into `CURVES`.
        nu (float | None): Student-t degrees of freedom for the added noise. Defaults to `NU`.

    Returns:
        tuple: `((med, lo, hi), (med, lo, hi))`, yield first then reading. The reading median is
            discarded at every call site.
    """
    nu = NU if nu is None else nu
    f = CURVES[curve]
    exp = harvest_scale(post, c_param) * f(day, post["A"], post["k"], post["t0"])
    sig_h = np.asarray(post.get("sigma_h", np.nan), float)
    noise = sig_h * noise_rng(*rng_key).standard_t(nu, size=np.shape(exp))

    return q95(exp), q95(np.clip(exp + noise, 0, None))


def sample_interval(post, day, curve=CURVE):
    """
    95% range for the sample-scale curve, `f(day)`.

    What a one-foot hand sample would read, with no `c` and no noise. Plotted beside the harvest
    band, where the gap between the two is `c`.

    Args:
        post (dict): a posterior with A, k, t0.
        day (float): day of season.
        curve (str): key into `CURVES`.

    Returns:
        tuple[float, float, float]: `(median, lo95, hi95)` in lbs/ft.
    """
    return q95(CURVES[curve](day, post["A"], post["k"], post["t0"]))


def naive_lbs_ft(seen, day):
    """
    GreenWave's current approach: carry forward the most recent reading.

    The baseline every score is measured against. When several observations land
    on that same most-recent day they are AVERAGED -- note this averages across
    scales where a sample and a harvest were both logged that day, which happens
    in 27 (farm-season, day) pairs and mixes a one-foot reading with a
    long-line one.

    Args:
        seen (list[tuple]): observations, as from `preprocess.season_events`.
        day (float): the day being forecast. Only strictly-earlier readings are
            used.

    Returns:
        float: the carried-forward lbs/ft. 0.0 when nothing precedes `day`:
            every such harvest in this dataset has an outplant logged before
            it, whose reading is exactly 0.00, and GreenWave's running figure
            before the first sample is 0. 14 of 325 scored harvests hit this
            case; they used to be dropped, which made a season's `actual_lbs`
            stop being its real harvested total.
    """
    prior_obs = [(d, v) for d, v, *_ in seen if d < day]

    if not prior_obs:
        # No sample or harvest yet, so GreenWave's running figure is 0 -- which
        # is also literally the last logged reading, since the outplant row
        # that precedes every one of these cases records 0.00 lbs/ft. Returning
        # NaN here used to drop the harvest from the scorecard entirely, which
        # made `actual_lbs` stop being the season's real harvested total.
        return 0.0

    last = max(d for d, _ in prior_obs)

    return float(np.mean([v for d, v in prior_obs if d == last]))


def point_estimate(post, day, point="median", c_param=C_PARAM, curve=CURVE):
    """
    One predicted sample and harvest lbs/ft for `day`.

    The median is the median of the predicted values, which minimises expected absolute error and so
    is right whenever the score is MAE. It ignores any relationship between the parameters; see
    `joint_mode_prediction` for the alternative and why it is rarely wanted.

    Args:
        post (dict): a posterior with A, k, t0, c.
        day (float): day of season.
        point (str): "median" or "mean". Defaults to "median".
        c_param (str): harvest-scale parameterisation.
        curve (str): key into `CURVES`.

    Returns:
        tuple[float, float]: `(sample_lbs_ft, harvest_lbs_ft)`. The harvest value is NaN when `c` is
            entirely non-finite, i.e. no harvests were seen.
    """
    f = CURVES[curve]
    agg = np.nanmedian if point == "median" else np.nanmean
    fd = f(day, post["A"], post["k"], post["t0"])
    hd = harvest_scale(post, c_param) * fd

    return float(agg(fd)), (float(agg(hd)) if np.isfinite(hd).any() else np.nan)


def joint_mode_prediction(post, day, c_param=C_PARAM, curve=CURVE):
    """
    The value at `day` of the single most probable curve.

    A Gaussian KDE over the draws picks the highest-density one, so the answer comes from one
    coherent parameter set rather than four marginals that may not co-occur. Needed only when
    coherence is the point; for scoring the median is correct, and backtests showed joint_mode
    performing worse.

    Args:
        post (dict): a posterior with A, k, t0, c.
        day (float): day of season.
        c_param (str): harvest-scale parameterisation.
        curve (str): key into `CURVES`.

    Returns:
        float: harvest lbs/ft at `day` from the modal draw. NaN if `c` is unidentified. Needs
            non-degenerate draws, since rank-1 draws give the KDE a singular covariance.
    """
    from scipy.stats import gaussian_kde

    sc = harvest_scale(post, c_param)
    arrs, names = [], []

    for name, v, logit in [
        ("A", post["A"], True),
        ("k", post["k"], True),
        ("t0", post["t0"], False),
        ("scale", sc, True),
    ]:
        if name == "scale" and not np.isfinite(v).any():
            continue
        arrs.append(np.log(v) if logit else v)
        names.append(name)

    X = np.vstack(arrs)
    X = X[:, np.isfinite(X).all(axis=0)]
    best = X[:, np.argmax(gaussian_kde(X)(X))]
    vals = {n: (float(np.exp(b)) if n != "t0" else float(b)) for n, b in zip(names, best)}
    f = CURVES[curve]

    return vals.get("scale", np.nan) * f(day, vals["A"], vals["k"], vals["t0"])


# --------------------------------------------------------------------------
# SeasonModel: the three things every call needs, bound together
# --------------------------------------------------------------------------
class SeasonModel:
    """
    Binds events, prior, and cache so prediction reads the way you think about it.

    Two independent axes, which the bare functions conflate. Conditioning is which observations the
    posterior was allowed to see, said as a date, a day of season, or a count. The target day is the
    day you want a number for, which need not be a day anything was observed. So "the forecast as of
    3 May, for 20 May" is one call, and nothing stops the target being in the future.
    """

    def __init__(self, events, prior_for, cache, sampler=None, line_events=None):
        """
        Bind the events, prior, and cache a prediction needs.

        Args:
            events (pd.DataFrame): events table with outplants, samples, and harvests.
            prior_for (callable): Mapping of fsid to its `Prior`.
            cache (PosteriorCache): supplies and stores posteriors.
            sampler (dict | None): sampler settings overrides, forwarded to the cache.
            line_events (pd.DataFrame | None): from `preprocess.load_line_events`; needed for the
                pounds views in `report`.
        """
        self.events = events
        self.prior_for = prior_for
        self.cache = cache
        self.sampler = sampler
        self.line_events = line_events

    # -- data ------------------------------------------------------------
    def observations(self, fsid):
        """
        That farm-season's samples and harvests in time order, indexed 1..N.

        The index is what `n_obs=` counts, so this is how to see what "the first 19 observations"
        means for a farm-season. Outplants are absent, since they are never conditioned on.

        Args:
            fsid (str): farm-season id.

        Returns:
            pd.DataFrame: indexed by `n`, with the date, day, event, lbs/ft, line_ft, and weight.
        """
        obs = season_events(self.events, fsid)
        g = self.events[self.events[IDCOL] == fsid]
        date_of = dict(zip(g[TCOL].astype(float), pd.to_datetime(g["Log Date"]).dt.date))
        t = pd.DataFrame(
            [
                {
                    "n": i + 1,
                    "date": date_of.get(d),
                    "day": d,
                    "event": k,
                    "lbs_ft": v,
                    "line_ft": lf,
                    "weight_lbs": w,
                }
                for i, (d, v, k, lf, w) in enumerate(obs)
            ]
        )

        return t.set_index("n") if len(t) else t

    def _day_of(self, fsid, as_of):
        """
        Resolve `as_of` to a day of season.

        A number passes through. A date is looked up among the farm-season's logged dates, and if
        nothing was logged that day it is interpolated from the season's start, so "as of 12 May"
        works whether or not anything happened then.

        Args:
            fsid (str): farm-season id.
            as_of (int | float | str | pd.Timestamp): a day of season or a date.

        Returns:
            float: the day of season.
        """
        if isinstance(as_of, int | float | np.integer | np.floating):
            return float(as_of)

        g = self.events[self.events[IDCOL] == fsid]
        target = pd.Timestamp(as_of).normalize()
        days = g.assign(_d=pd.to_datetime(g["Log Date"]).dt.normalize())
        hit = days[days["_d"] == target]

        if len(hit):
            return float(hit[TCOL].iloc[0])

        # a date with nothing logged on it: interpolate from the season's start
        start = days["_d"].min() - pd.Timedelta(days=float(days[TCOL].min()))

        return float((target - start).days)

    # -- posteriors ------------------------------------------------------
    def posterior(self, fsid, as_of=None, n_obs=None, inclusive=False):
        """
        The posterior conditioned on part of a farm-season's history.

        `as_of` is strictly before by default: `as_of="12 May"` sees everything logged on 11 May and
        earlier, and nothing from the 12th. Inclusive would let a harvest inform its own forecast,
        the leak every table here avoids, so it has to be asked for explicitly.

        Args:
            fsid (str): farm-season id.
            as_of (int | float | str | pd.Timestamp | None): condition on everything before this day
                or date. Exactly one of this and `n_obs`.
            n_obs (int | None): condition on the first `n_obs` observations instead, in date order.
                Clamped to the number available.
            inclusive (bool): include observations on `as_of`. Defaults to False.

        Returns:
            dict: the posterior, plus the farm-season, how many it saw, and the last day it saw, so
                `predict` can report the extrapolation gap.
        """
        if (as_of is None) == (n_obs is None):
            raise ValueError("give exactly one of as_of= or n_obs=")

        obs = season_events(self.events, fsid)

        if n_obs is None:
            day = self._day_of(fsid, as_of)
            n_obs = sum(1 for d, *_ in obs if (d <= day if inclusive else d < day))

        n_obs = int(min(n_obs, len(obs)))
        post = dict(self.cache.get(fsid, obs, n_obs, self.prior_for(fsid), self.sampler))
        # provenance, so predict() can report the extrapolation gap
        post["_fsid"] = fsid
        post["_n_seen"] = n_obs
        post["_last_day"] = obs[n_obs - 1][0] if n_obs else np.nan

        return post

    # -- prediction ------------------------------------------------------
    def predict(self, post, day, estimator="median", line_ft=None):
        """
        One forecast for one day, from one posterior.

        The target day is independent of what the posterior saw, so it need not be a day anything
        was logged and may be in the future.

        `estimator` is how 4,000 draws become one number. The median of the predicted values
        minimises expected absolute error and is the right default; the mean minimises squared
        error; joint_mode takes the single most probable curve, needed only when a coherent
        parameter set matters.

        Args:
            post (dict): a posterior, normally from `posterior`.
            day (float): the target day of season.
            estimator (str): as above. Defaults to "median".
            line_ft (float | None): feet of line. Given, the pounds columns are added. The choice of
                footage is the caller's, since "what this cut will weigh" and "what is standing" use
                different ones.

        Returns:
            pd.Series: the sample and harvest lbs/ft, both 95% ranges, the diagnostics, and
                `gap_days`, the distance from the last observation seen to the target. A large gap
                is why a range is wide: the forecast is extrapolating along a curve the data has not
                checked out there.
        """
        c_param = C_PARAM
        (y_med, y_lo, y_hi), (_, r_lo, r_hi) = harvest_intervals(
            post,
            day,
            (post.get("_fsid", "-"), day, post.get("_n_seen", 0), "predict"),
            c_param,
            self.cache.curve,
        )

        s_med, s_lo, s_hi = sample_interval(post, day, self.cache.curve)

        if estimator == "joint_mode":
            y_med = joint_mode_prediction(post, day, c_param, self.cache.curve)
        elif estimator == "mean":
            s_med, y_med = point_estimate(post, day, "mean", c_param, self.cache.curve)

        out = {
            "farm_season_id": post.get("_fsid"),
            "day": float(day),
            "estimator": estimator,
            "n_obs_seen": post.get("_n_seen"),
            "last_obs_day": post.get("_last_day"),
            "gap_days": float(day) - post.get("_last_day", np.nan),
            "sample_lbs_ft": s_med,
            "harvest_lbs_ft": y_med,
            "yield_lo95": y_lo,
            "yield_hi95": y_hi,
            "reading_lo95": r_lo,
            "reading_hi95": r_hi,
            # numeric, not a formatted string: a Series with a string in
            # it cannot be rounded or arithmetic'd
            "r_hat": post.get("rhat_max", np.nan),
            "ess": post.get("ess", np.nan),
        }

        if line_ft is not None:
            out.update(
                {
                    "line_ft": float(line_ft),
                    "lbs": y_med * line_ft,
                    "lbs_lo95": y_lo * line_ft,
                    "lbs_hi95": y_hi * line_ft,
                    "lbs_reading_lo95": r_lo * line_ft,
                    "lbs_reading_hi95": r_hi * line_ft,
                }
            )

        return pd.Series(out)

    def predict_at(
        self, fsid, day, as_of=None, n_obs=None, estimator="median", line_ft=None, inclusive=False
    ):
        """
        `posterior` then `predict` in one call.

        Args:
            fsid (str): farm-season id.
            day (float): the target day.
            as_of (int | float | str | pd.Timestamp | None): conditioning cutoff. With neither this
                nor `n_obs`, conditions on everything strictly before `day`, the honest running
                forecast.
            n_obs (int | None): condition on a count instead.
            estimator (str): see `predict`.
            line_ft (float | None): see `predict`.
            inclusive (bool): see `posterior`.

        Returns:
            pd.Series: as `predict`.
        """
        if as_of is None and n_obs is None:
            as_of = day

        post = self.posterior(fsid, as_of=as_of, n_obs=n_obs, inclusive=inclusive)

        return self.predict(post, day, estimator=estimator, line_ft=line_ft)

    # -- what's cached ---------------------------------------------------
    def index(self):
        """
        Every posterior in the cache, in terms you can look up.

        So "the posterior for Farm_2 as of day 224" becomes a lookup rather than an observation
        count you have to work out yourself.

        Returns:
            pd.DataFrame: one row per cached entry with the farm-season, how many it saw, the day it
                saw up to, the sample and harvest counts, the prior source, and the diagnostics.
        """
        rows = []
        obs_cache = {}

        for k, v in self.cache.store.items():
            if not (isinstance(k, tuple) and len(k) > _KEY_PRIOR_FINGERPRINT):
                continue
            fsid, n_seen = k[0], k[1]
            if fsid not in obs_cache:
                try:
                    obs_cache[fsid] = season_events(self.events, fsid)
                except Exception:
                    obs_cache[fsid] = []
            obs = obs_cache[fsid]
            seen = obs[:n_seen]
            rows.append(
                {
                    "farm_season_id": fsid,
                    "n_seen": n_seen,
                    "as_of_day": seen[-1][0] if seen else np.nan,
                    "n_samples": sum(1 for e in seen if e[2] == SAMPLE_EVENT),
                    "n_harvests": sum(1 for e in seen if e[2] == HARVEST_EVENT),
                    "prior_source": k[_KEY_PRIOR_SOURCE],
                    "method": v.get("method", "?"),
                    "r_hat": v.get("rhat_max", np.nan),
                    "ess": v.get("ess", np.nan),
                }
            )

        t = pd.DataFrame(rows)

        return t.sort_values(["farm_season_id", "n_seen"]).reset_index(drop=True)

    # -- the per-season replay ------------------------------------------
    def report(self, fsid, **kw):
        """
        `evaluate.season_report` with this model's events, prior, cache, and sampler.

        Args:
            fsid (str): farm-season id to replay.
            **kw: forwarded to `evaluate.season_report`.

        Returns:
            pd.DataFrame: the season replay.
        """
        from .evaluate import season_report  # late: evaluate imports model

        return season_report(
            self.events,
            fsid,
            self.prior_for,
            self.cache,
            sampler=self.sampler,
            line_events=self.line_events,
            **kw,
        )


# --------------------------------------------------------------------------
# sampler diagnostics
# --------------------------------------------------------------------------
def posterior_diagnostics(cache, fsids=None):
    """
    One row per cached posterior: did the sampler actually work?

    Three things to look at, in order of how much they should worry you. Divergences mean a
    transition the integrator could not follow, so part of the posterior was never explored and the
    draws are biased, not just noisy. `r_hat` compares between-chain to within-chain variance,
    rank-normalised and split so it also catches one chain drifting within itself; above 1.01 the
    chains have not agreed yet. `ess` is the effective sample size, and below ~400 the quantiles are
    unreliable. Note `ess` here is the minimum across parameters; see `forecast_mc_error` for the
    forecast's own effective sample size, which is usually much better.

    Prior-only rows carry no diagnostics and are reported separately rather than counted as passes.

    Args:
        cache (PosteriorCache): the store to inspect.
        fsids (list[str] | None): restrict to these farm-seasons. Worth using, since pooling across
            sampler settings mixes a short exploratory run in with a production one.

    Returns:
        pd.DataFrame: one row per posterior, with `flag` naming the single worst problem per row.
    """
    rows = []

    for k, v in cache.store.items():
        fsid, n_seen = (k[0], k[1]) if isinstance(k, tuple) else ("?", None)
        if fsids is not None and fsid not in fsids:
            continue
        rows.append(
            {
                "farm_season_id": fsid,
                "n_seen": n_seen,
                "method": v.get("method", "?"),
                "r_hat": v.get("rhat_max", np.nan),
                "ess": v.get("ess", np.nan),
                "divergences": v.get("divergences", np.nan),
                "prior_source": (
                    k[_KEY_PRIOR_SOURCE]
                    if isinstance(k, tuple) and len(k) > _KEY_PRIOR_SOURCE
                    else "?"
                ),
            }
        )

    t = pd.DataFrame(rows)

    if len(t):
        t["flag"] = np.where(
            t["divergences"].fillna(0) > 0,
            "divergences",
            np.where(
                t["r_hat"] > RHAT_OK,
                f"r_hat>{RHAT_OK}",
                np.where(t["ess"] < ESS_MIN, f"ess<{ESS_MIN}", ""),
            ),
        )

    return t


def diagnostics_summary(diag, quiet=False):
    """
    Headline pass or fail over a `posterior_diagnostics` table.

    Scores only the mcmc rows; prior-only rows have nothing to converge and are counted separately.

    Args:
        diag (pd.DataFrame): `posterior_diagnostics` output.
        quiet (bool): return the numbers without printing them. Defaults to False, which also prints
            the five worst offenders by ess.

    Returns:
        dict: the counts and percentages. Empty if nothing was fitted.
    """
    fitted = diag[diag["method"] == "mcmc"]

    if not len(fitted):
        return {}

    out = {
        "posteriors fitted": len(fitted),
        "prior-only rows": int((diag["method"] == "prior").sum()),
        f"% r_hat <= {RHAT_OK}": 100 * (fitted["r_hat"] <= RHAT_OK).mean(),
        f"% r_hat <= {RHAT_LOOSE}": 100 * (fitted["r_hat"] <= RHAT_LOOSE).mean(),
        "worst r_hat": fitted["r_hat"].max(),
        "median ess": fitted["ess"].median(),
        "min ess": fitted["ess"].min(),
        f"n with ess < {ESS_MIN}": int((fitted["ess"] < ESS_MIN).sum()),
        "n with divergences": int((fitted["divergences"].fillna(0) > 0).sum()),
    }

    if not quiet:
        for k, v in out.items():
            print(f"  {k:22s} {v:,.1f}" if isinstance(v, float) else f"  {k:22s} {v:,}")
        worst = fitted[fitted["flag"] != ""].sort_values("ess")
        if len(worst):
            print(f"  worst offenders (of {len(worst)} flagged):")
            print(
                worst.head(5)[
                    ["farm_season_id", "n_seen", "r_hat", "ess", "divergences", "flag"]
                ].to_string(index=False)
            )

    return out


def forecast_mc_error(post, day, c_param=C_PARAM, curve=CURVE, probs=(0.025, 0.5, 0.975)):
    """
    Monte Carlo error on the numbers a report actually quotes.

    Not the distance to the true posterior. It is how much the reported figure would move if the
    sampler were re-run with another seed. Every reported number is an integral approximated by
    averaging over the draws, and this is the standard error of that average.

    Read it against `posterior_sd` beside it. The sd is genuine uncertainty about the yield and
    shrinks only with more data; the mcse is arithmetic noise and shrinks as `1/sqrt(draws)`. When
    mcse is a few percent of the sd, sampling is not what limits the answer, and quoting more
    decimals than the mcse is false precision.

    `posterior_sd / mcse_mean` squared is the forecast's own effective sample size, typically far
    better than the minimum parameter ess, because `A` and `t0` are strongly correlated so each
    wanders while the combination they produce stays pinned. The quantile mcses are larger than the
    mean's and the tails largest of all, which is why an in95 flag can flip between seeds for a
    harvest on the boundary.

    The chain count is not recorded by `posterior_mcmc`, so this assumes 4. With a cache sampled at
    a different `chains` the reshape splits chains rather than separating them, making the estimate
    behave like a split-chain one.

    Args:
        post (dict): a posterior with A, k, t0, c.
        day (float): the day the quoted forecast is for.
        c_param (str): harvest-scale parameterisation.
        curve (str): key into `CURVES`.
        probs (tuple[float, ...]): quantiles to report mcse for. Defaults to the three a report
            quotes.

    Returns:
        dict: `posterior_sd`, `mcse_mean`, one `mcse_q` per entry in `probs`, and `ess`.
    """
    import arviz as az

    f = CURVES[curve]
    g = harvest_scale(post, c_param) * f(day, post["A"], post["k"], post["t0"])
    chains = max(int(post.get("n_chains", 4)), 1)
    g2 = g.reshape(chains, -1) if g.size % chains == 0 else g.reshape(1, -1)
    out = {"posterior_sd": float(np.std(g)), "mcse_mean": float(az.mcse(g2, method="mean"))}

    for p in probs:
        out[f"mcse_q{p:g}"] = float(az.mcse(g2, method="quantile", prob=p))

    out["ess"] = float(post.get("ess", np.nan))

    return out


def sampler_tag(sampler=None):
    """
    Short, sortable label for a sampler configuration.

    Appears in cache filenames, so two configurations cannot land on the same file.

    Args:
        sampler (dict | None): sampler settings. Merged with `SAMPLER_DEFAULTS`.

    Returns:
        str: e.g. "d1000_t1000_c4_ta90_s0" -- draws, tune, chains, target_accept, seed.
    """
    s = {**SAMPLER_DEFAULTS, **(sampler or {})}

    return (
        f"d{s['draws']}_t{s['tune']}_c{s['chains']}"
        f"_ta{int(round(s['target_accept'] * 100))}_s{s['seed']}"
    )


def cache_name(prefix, label="", sampler=None, data_meta=None, prior=None):
    """
    A filename that says what is inside it.

    Built from everything that determines the contents, so two runs cannot land on the same file and
    silently mix: `myrun_unpooled_p28e2119b_d1000_t1000_c4_ta90_s0_x1964ba10`. Being explicit is
    cheaper than the alternative: a name like "post_cache" tells you nothing, and appending a second
    run's entries to it leaves a file whose contents you can only discover by loading it.

    Args:
        prefix (str): yours, whatever names the run for you. Everything after it is derived, so your
            label stays readable and the machine-checkable detail follows.
        label (str): which model made it, "unpooled" or "hierarchical".
        sampler (dict | None): becomes the sampler segment via `sampler_tag`.
        data_meta (dict | str | None): `load_dataset`'s meta, or a fingerprint directly. A different
            extract or filter set gets a different file.
        prior (Prior | str | None): becomes the prior segment. Pass None for the hierarchical model,
            whose prior differs per farm-season, and let `label` carry the identity.

    Returns:
        str: the name, without the ".pkl" extension.
    """
    parts = [str(prefix)]

    if label:
        parts.append(str(label))

    if prior is not None:
        fp = prior["_meta"].get("fingerprint") if hasattr(prior, "__getitem__") else prior
        parts.append(f"p{fp}")

    parts.append(sampler_tag(sampler))

    if data_meta is not None:
        fp = data_meta["fingerprint"] if isinstance(data_meta, dict) else data_meta
        parts.append(f"x{fp}")

    return "_".join(parts)
