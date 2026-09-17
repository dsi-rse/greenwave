from scipy.stats import median_abs_deviation
from scipy.optimize import least_squares
import numpy as np
import pandas as pd

from .curves import CURVES

YCOL = "lbs_ft"
TCOL = "day_of_season"
IDCOL = "farm_season_id"

SAMPLE_EVENT = "sample"      # value in `event` column for samples
OUTPLANT_EVENT = "outplant"  # value in `event` column for outplanting
HARVEST_EVENT = "harvest"    # value in `event` column for harvests
MIN_SAMPLES = 3              # skip seasons with fewer sample points than this


def fit_one_season_to_curve(t, y, th=None, yh=None, curve="logistic"):
    """
    Fit one farm-season to provided curve function.

    Curve function must be mapped to string in ``CURVES``.
    (t, y) are the sample observations. (th, yh) are the harvest observations.

    Includes initialization of curve parameters based on data and previous work.

    """
    f = CURVES[curve]
    t, y = np.asarray(t, float), np.asarray(y, float)

    # separate harvests from samples. On different scales due to measurement differences
    has_h = th is not None and yh is not None and len(np.atleast_1d(th)) > 0
    if has_h:
        th, yh = np.asarray(th, float), np.asarray(yh, float)
    
    # Data driven initialization of curve parameters

    # A0 guess: plateau is 1.5*max sample yield
    # samples are usually taken before harvests, and thus are lacking in late season
    ymax = y.max()
    A0 = ymax * 1.5 
    # t0 guess: day of steepest observed increase in y is the median amongst all samples
    order = np.argsort(t)
    t_sorted, y_sorted = t[order], y[order]
    t0_0 = float(np.median(t_sorted))

    # k guess: growth rate parameter is assumed to be small for lbs/ft/day
    k0 = 0.05 
    # harvest/sample scale guess: from fake bayesian's scaling factor
    c0 = 0.65 

    # Bounds for parameters. Answers what's realistic and what we want to flag if not
    # A must be positive and probably not 3x outside all observations
    # k must be positive for growth rate, likely not more than 1 lb/ft of growth per day
    # t0 is likely not 150 days before the first observation and 150 days after the last observation
    lo = [1e-3, 1e-4, t_sorted.min()-150]
    hi = [3.0 * max(ymax, 0.1), 1.0, t_sorted.max()+150]
    x0 = [A0, k0, t0_0]
    if has_h: # add c bounds, likely not 0.1x samples and not 3x the samples
        lo, hi, x0 = lo + [0.1], hi + [3.0], x0 + [c0]
    x0 = np.clip(x0, lo, hi) # ensures the guesses are within the bounds
    # `soft_l1` is used to calculate the loss in the least_squares optimizer 
    # when fitting the curve to farm-season
    # `soft_l1` is robust to extreme residuals by capping their inflence with a f_scale boundary
    # f_scale is the boundary where quadratic loss for small residuals, linear loss for large
    f_scale = max(0.1 * ymax, 1e-3)



    def residuals(p):
        """
        Calculates the residual between the model's prediction and the observed yield.

        Calculates a harvest and sample residual if harvest is present in the farm-season.
        """
        if has_h:
            A_, k_, t0_, c_ = p
            return np.concatenate([f(t, A_, k_, t0_) - y,
                                   c_ * f(th, A_, k_, t0_) - yh])
        return f(t, *p) - y

    # fit curve to farm-season, calculate residuals and sigma based on those residuals
    try:
        # fit the data to a logistic curve, report winning parameters
        res = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1",
                            f_scale=f_scale, max_nfev=5000)
        if has_h: # calculate residual for harvest curve
            A, k, t0, c = res.x
            rh = c * f(th, A, k, t0) - yh 
        else: # just sample residual calculated, no harvest c factor nor residual
            (A, k, t0), c, rh = res.x, np.nan, np.array([]) 
        r = f(t, A, k, t0) - y  # sample residuals only -> sample-scale sigma


        # Calculate sigma - typical data scatter around curve, measurement noise/uncertainty
        # Use median absolute deviation which is robust to outliers
        # 1. takes the median residual and subtracts it from each residual
        # 2. take the median of the absolute value of #1's result
        # 3. scale by 1.4826 to convert to Gaussian's noise scale
        sigma = median_abs_deviation(r, scale='normal')
        # Only calculate harvest's sigma if more than 2 harvest, else nan, can't determine deviation
        sigma_h = (median_abs_deviation(rh, scale='normal')) if len(rh) > 2 else np.nan
        rmse = float(np.sqrt(np.mean(r ** 2)))
        converged = bool(res.success)
    except Exception:
        # if an exception occurred, set all parameters to NaN
        A = k = t0 = c = sigma = sigma_h = rmse = np.nan
        converged = False

    # Quality diagnostics that Claude came up with... need further investigation but kept
    # to be a placeholder for evaluating whether a parameter set should be included in the prior
    # --- identifiability / quality diagnostics -----------------------------
    # Plateau is informed if any observations (samples or harvests) continued
    # past t0 + 1/k where the logistic curve reaches ~73% of its plateau
    t_obs_max = max(t_sorted.max(), th.max()) if has_h else t_sorted.max()
    plateau_seen = bool(np.isfinite(t0) and
                        t_obs_max > t0 + (1.0 / max(k, 1e-6)))
    # A far above anything observed => extrapolated plateau. Judge this on
    # the SAMPLE scale only: converting harvests via the fitted c would be
    # circular (a degenerate c would launder a degenerate A past the check).
    a_ratio = float(A / ymax) if np.isfinite(A) and ymax > 0 else np.nan
 
    # t0 pinned at the edge of its allowed window = optimizer ran away,
    # a classic sign the sigmoid shape isn't identified by this season.
    t0_at_bound = bool(np.isfinite(t0) and
                       (t0 <= lo[2] + 1 or t0 >= hi[2] - 1))
    # c pinned at its bound = scale not identified (usually: harvests are
    # the only late-season evidence, so c and A trade off).
    c_at_bound = bool(has_h and np.isfinite(c) and
                      (c <= 0.1 + 1e-3 or c >= 3.0 - 1e-3))
    # The A-c split is only trustworthy if SAMPLES pinned the curve's bend;
    # harvests alone can't separate "high plateau, low c" from the reverse.
    samples_saw_bend = bool(np.isfinite(t0) and
                            t_sorted.max() > t0 + (1.0 / max(k, 1e-6)))
 
    ok = (converged and np.isfinite(a_ratio) and not t0_at_bound
          and not c_at_bound and len(y) >= MIN_SAMPLES
          and (samples_saw_bend or a_ratio < 2.0))
    return dict(A=A, k=k, t0=t0, c=c, sigma=sigma, sigma_h=sigma_h,
                rmse=rmse, n_samples=len(y),
                n_harvests=int(len(yh)) if has_h else 0,
                converged=converged, plateau_seen=plateau_seen,
                samples_saw_bend=samples_saw_bend,
                A_over_ymax=a_ratio, t0_at_bound=t0_at_bound,
                c_at_bound=c_at_bound,
                quality=("ok" if ok else "flag"))

def build_parameter_cloud(df, curve="logistic"):
    """
    Fit every farm-season; return one tagged row per season.

    Samples + outplanting events split from harvest events. Have different measurement error.
    Groups events by same farm-season, drops farm seasons with less than 3 sample events,
    and fits a curve. Records the fitted curve's parameters, quality flags, + group metadata.
    """
    samples = df[(df["event"] == SAMPLE_EVENT) | (df["event"] == OUTPLANT_EVENT)].dropna(subset=[YCOL, TCOL])
    harvests = df[df["event"] == HARVEST_EVENT].dropna(subset=[YCOL, TCOL])
    rows = []
    for fsid, g in samples.groupby(IDCOL):
        if len(g) < MIN_SAMPLES:
            continue
        h = harvests[harvests[IDCOL] == fsid]
        fit = fit_one_season_to_curve(g[TCOL], g[YCOL],
                             th=h[TCOL].values if len(h) else None,
                             yh=h[YCOL].values if len(h) else None,
                             curve=curve)
        meta = g.iloc[0]
        fit.update({
            IDCOL: fsid,
            "farm": meta["Farm Name"],
            "season": meta["Season"],
            "state": meta["State"],
            "outplant_spread_days": meta.get("outplant_spread_days", np.nan),
            "n_outplant_dates": meta.get("n_outplant_dates", np.nan),
            "single_outplant": meta.get("flag_single_outplant", np.nan),
            "curve": curve,
        })
        rows.append(fit)
    cloud = pd.DataFrame(rows)
    return cloud.sort_values("rmse", ascending=False).reset_index(drop=True)