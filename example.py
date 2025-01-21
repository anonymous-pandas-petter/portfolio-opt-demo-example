# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "cvxpy",
#   "tqdm",
#   "scipy",
#   "pyarrow"
# ]
# ///

import argparse

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.linalg
from tqdm import tqdm


def optimize_quadform(
    historical_returns: pd.DataFrame,
    window_size: pd.Timedelta,
    gamma: float = 0.25,
):
    if not isinstance(historical_returns.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    n_assets = historical_returns.shape[1]

    mu = cp.Parameter(n_assets)  # expected returns
    cov_m = cp.Parameter((n_assets, n_assets))  # covariance
    gamma_const = cp.Parameter(nonneg=True)

    w_pos = cp.Variable(n_assets, nonneg=True)
    w_neg = cp.Variable(n_assets, nonneg=True)

    constraints = [
        cp.sum(w_pos + w_neg) == 1,
    ]

    w = w_pos - w_neg

    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(cov_m)))

    problem = cp.Problem(objective, constraints)

    is_dpp, is_dcp = (
        objective.is_dcp(dpp=True),
        objective.is_dcp(dpp=False),
    )

    print(f"[Quadform] Objective: DPP -> {is_dpp}, DCP -> {is_dcp}")

    gamma_const.value = gamma

    weights = []

    start_date = historical_returns.index.min() + window_size
    end_date = historical_returns.index.max() - pd.Timedelta(hours=24)

    for day in tqdm(pd.date_range(start_date, end_date, freq="1D")):
        window_end = day
        train_data = historical_returns[window_end - window_size : window_end]

        mu.value = train_data.mean(axis=0).to_numpy()
        cov_matrix = np.cov(train_data, rowvar=False)
        cov_m.value = cov_matrix

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        computed_weights = (day + pd.DateOffset(days=1), w.value)

        weights.append(computed_weights)

    df = (
        pd.DataFrame(
            np.array([w for _, w in weights]),
            columns=historical_returns.columns,
            index=[d for d, _ in weights],
        )
        .resample("h")
        .ffill()
    )

    df.index.name = historical_returns.index.name

    return df


def optimize_cholesky(
    historical_returns: pd.DataFrame,
    window_size: pd.Timedelta,
    gamma: float = 0.25,
):
    if not isinstance(historical_returns.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    n_assets = historical_returns.shape[1]

    mu = cp.Parameter(n_assets)  # expected returns
    cholesky_param = cp.Parameter((n_assets, n_assets))  # covariance

    w_pos = cp.Variable(n_assets, nonneg=True)
    w_neg = cp.Variable(n_assets, nonneg=True)

    constraints = [
        cp.sum(w_pos + w_neg) == 1,
    ]

    w = w_pos - w_neg

    objective = cp.Maximize(mu @ w - gamma * cp.sum_squares(cholesky_param @ w))

    problem = cp.Problem(objective, constraints)

    is_dpp, is_dcp = (
        objective.is_dcp(dpp=True),
        objective.is_dcp(dpp=False),
    )

    print(f"[Cholesky] Objective: DPP -> {is_dpp}, DCP -> {is_dcp}")

    weights = []

    start_date = historical_returns.index.min() + window_size
    end_date = historical_returns.index.max() - pd.Timedelta(hours=24)

    for day in tqdm(pd.date_range(start_date, end_date, freq="1D")):
        window_end = day
        train_data = historical_returns[window_end - window_size : window_end]

        mu.value = train_data.mean(axis=0).to_numpy()
        cov_matrix = np.cov(train_data, rowvar=False)

        # Ensure PSD by adding jitter if necessary
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if np.min(eigvals) < 1e-6:
            cov_matrix += np.eye(n_assets) * (abs(np.min(eigvals)) + 1e-6)
        cholesky_param.value = scipy.linalg.cholesky(cov_matrix)

        problem.solve(solver=cp.SCS, warm_start=True, enforce_dpp=True)

        computed_weights = (day + pd.DateOffset(days=1), w.value)

        weights.append(computed_weights)

    df = (
        pd.DataFrame(
            np.array([w for _, w in weights]),
            columns=historical_returns.columns,
            index=[d for d, _ in weights],
        )
        .resample("h")
        .ffill()
    )

    df.index.name = historical_returns.index.name

    return df


def main():
    window_size = pd.Timedelta(days=30)
    gamma = 0.08

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-data-points", "-n", type=int, default=2000)
    args = parser.parse_args()

    print("Reading data from file...")
    data = pd.read_parquet("asset_data.parquet")

    print("Number of assets:", data.shape[1])
    print("Computing weights using quadform...")
    weights_quadform = optimize_quadform(
        data[: args.n_data_points],
        window_size=window_size,
        gamma=gamma,
    )

    print("Computing weights using cholesky...")
    weights_cholesky = optimize_cholesky(
        data[: args.n_data_points],
        window_size=window_size,
        gamma=gamma,
    )


if __name__ == "__main__":
    main()
