"""Learning-curve power-law fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def power_law(n, c, alpha, epsilon):
    return c * np.power(n, -alpha) + epsilon


@dataclass
class PowerLawFit:
    c: float
    alpha: float
    epsilon: float
    epsilon_covariance_se: float
    epsilon_ci_low: float
    epsilon_ci_high: float
    bootstrap_fail_rate: float
    status: str


def fit_curve(n_samples: np.ndarray, mse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n_samples, dtype=float)
    y = np.asarray(mse, dtype=float)
    p0 = [max(y) - min(y), 0.5, max(0.0, min(y) * 0.9)]
    bounds = ([0.0, 1e-8, 0.0], [np.inf, 5.0, np.inf])
    params, covariance = curve_fit(
        power_law,
        n,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    return params, covariance


def fit_power_law_for_model(
    group: pd.DataFrame,
    null_mse: float,
    seed: int = 333,
    bootstrap_iterations: int = 2000,
) -> PowerLawFit:
    curve = group.groupby("n_samples", as_index=False)["mse"].mean().sort_values("n_samples")
    params, covariance = fit_curve(
        curve["n_samples"].to_numpy(dtype=float),
        curve["mse"].to_numpy(dtype=float),
    )
    c_hat, alpha_hat, epsilon_hat = [float(v) for v in params]
    cov_se = float(np.sqrt(np.diag(covariance))[2]) if covariance.size else float("nan")

    rng = np.random.default_rng(seed)
    boot_eps = []
    failed = 0
    grouped = {n: g["mse"].to_numpy(dtype=float) for n, g in group.groupby("n_samples")}
    ns = np.array(sorted(grouped), dtype=float)
    for _ in range(bootstrap_iterations):
        try:
            means = np.array(
                [rng.choice(grouped[int(n)], size=len(grouped[int(n)]), replace=True).mean() for n in ns],
                dtype=float,
            )
            boot_params, _ = fit_curve(ns, means)
            boot_eps.append(float(boot_params[2]))
        except Exception:
            failed += 1

    fail_rate = failed / max(1, bootstrap_iterations)
    if boot_eps:
        ci_low, ci_high = np.percentile(boot_eps, [2.5, 97.5])
    else:
        ci_low, ci_high = float("nan"), float("nan")

    status = "stable"
    if epsilon_hat < 0 or epsilon_hat > null_mse or alpha_hat <= 0 or fail_rate > 0.10:
        status = "unstable"

    return PowerLawFit(
        c=c_hat,
        alpha=alpha_hat,
        epsilon=epsilon_hat,
        epsilon_covariance_se=cov_se,
        epsilon_ci_low=float(ci_low),
        epsilon_ci_high=float(ci_high),
        bootstrap_fail_rate=float(fail_rate),
        status=status,
    )


def fit_all_power_laws(
    results: pd.DataFrame,
    null_mse: float,
    seed: int,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    rows = []
    for model, group in results.groupby("model"):
        try:
            fit = fit_power_law_for_model(
                group,
                null_mse=null_mse,
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
            )
            rows.append({"model": model, **fit.__dict__})
        except Exception as exc:
            rows.append(
                {
                    "model": model,
                    "c": np.nan,
                    "alpha": np.nan,
                    "epsilon": np.nan,
                    "epsilon_covariance_se": np.nan,
                    "epsilon_ci_low": np.nan,
                    "epsilon_ci_high": np.nan,
                    "bootstrap_fail_rate": 1.0,
                    "status": "fit_failed",
                    "failure_reason": str(exc),
                }
            )
    return pd.DataFrame(rows)
