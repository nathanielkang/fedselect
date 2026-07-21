"""Classical federated baselines on naive observed labels (Idea 1 E2)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .dgp import FederatedDataset
from .estimators import EstimateResult, _expand_design, _fit_outcome_regressor


def _client_design_xy(
    x: np.ndarray, y_obs: np.ndarray, observed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mask = observed
    if not mask.any():
        return np.zeros((0, 1)), np.zeros(0)
    x_design = _expand_design(x[mask])
    y = y_obs[mask]
    return x_design, y


def _x_all_matrix(data: FederatedDataset) -> np.ndarray:
    x_all = data.x_all
    if x_all.ndim == 1:
        return x_all.reshape(-1, 1)
    return x_all


def _feature_dim(data: FederatedDataset) -> int:
    x_all = _x_all_matrix(data)
    return _expand_design(x_all[:1]).shape[1]


def _fedavg_style_mean_prediction(
    avg_coef: np.ndarray, avg_intercept: float, data: FederatedDataset
) -> float:
    """Mean prediction on population support using expanded-design coefficients."""
    x_all = _x_all_matrix(data)
    x_design = _expand_design(x_all)
    preds = x_design @ avg_coef[: x_design.shape[1]] + avg_intercept
    return float(np.mean(preds))


def _proximal_ridge_fit(
    x_design: np.ndarray,
    y: np.ndarray,
    global_coef: np.ndarray,
    global_intercept: float,
    mu: float,
) -> Tuple[np.ndarray, float]:
    """Closed-form FedProx local step on expanded design."""
    n_feat = x_design.shape[1]
    x_aug = np.hstack([x_design, np.ones((x_design.shape[0], 1))])
    w_global = np.concatenate([global_coef, [global_intercept]])
    gram = x_aug.T @ x_aug + mu * np.eye(n_feat + 1)
    rhs = x_aug.T @ y + mu * w_global
    w = np.linalg.solve(gram, rhs)
    return w[:-1], float(w[-1])


def estimate_fedprox(
    data: FederatedDataset,
    mu: float = 0.1,
    n_rounds: int = 2,
) -> EstimateResult:
    """
    FedProx on naive observed labels: each client fits a ridge-style head on
    censored outcomes with proximal pull toward the global model (Li et al., MLSys 2020).
    """
    n_feat = _feature_dim(data)
    global_coef = np.zeros(n_feat)
    global_intercept = 0.0
    n_total = 0

    for _ in range(max(n_rounds, 1)):
        coefs: List[np.ndarray] = []
        intercepts: List[float] = []
        weights: List[int] = []
        for c in data.clients:
            x_design, y = _client_design_xy(c.x, c.y_obs, c.observed)
            if x_design.shape[0] == 0:
                continue
            coef, intercept = _proximal_ridge_fit(
                x_design, y, global_coef, global_intercept, mu=mu
            )
            coefs.append(coef)
            intercepts.append(intercept)
            weights.append(x_design.shape[0])
            n_total += x_design.shape[0]

        if not coefs:
            return EstimateResult(method="fedprox", estimate=float("nan"), n_moments=0)

        w_arr = np.array(weights, dtype=float)
        global_coef = np.average(np.vstack(coefs), axis=0, weights=w_arr)
        global_intercept = float(np.average(intercepts, weights=w_arr))

    est = _fedavg_style_mean_prediction(global_coef, global_intercept, data)
    return EstimateResult(
        method="fedprox",
        estimate=est,
        n_moments=n_total,
    )


def _local_grad(
    x_design: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: float
) -> Tuple[np.ndarray, float]:
    """Gradient of 1/(2n) MSE w.r.t. (coef, intercept) at current point."""
    n = x_design.shape[0]
    if n == 0:
        return np.zeros_like(coef), 0.0
    x_aug = np.hstack([x_design, np.ones((n, 1))])
    w = np.concatenate([coef, [intercept]])
    pred = x_aug @ w
    resid = pred - y
    grad_w = (x_aug.T @ resid) / n
    return grad_w[:-1], float(grad_w[-1])


def estimate_scaffold(
    data: FederatedDataset,
    n_rounds: int = 2,
    lr: float = 0.5,
) -> EstimateResult:
    """
    Lightweight SCAFFOLD on the same naive-label regression objective
    (Karimireddy et al., ICML 2020). Control variates correct client drift
    on expanded-design MSE gradients.
    """
    n_feat = _feature_dim(data)
    global_coef = np.zeros(n_feat)
    global_intercept = 0.0
    c_coef = np.zeros(n_feat)
    c_intercept = 0.0
    client_c: List[Tuple[np.ndarray, float]] = [
        (np.zeros(n_feat), 0.0) for _ in data.clients
    ]
    n_total = 0

    for _ in range(max(n_rounds, 1)):
        new_coefs: List[np.ndarray] = []
        new_intercepts: List[float] = []
        weights: List[int] = []
        delta_c: List[Tuple[np.ndarray, float]] = []

        for cid, c in enumerate(data.clients):
            x_design, y = _client_design_xy(c.x, c.y_obs, c.observed)
            c_k_coef, c_k_intercept = client_c[cid]
            g_coef, g_intercept = _local_grad(
                x_design, y, global_coef, global_intercept
            )
            step_coef = lr * (g_coef - c_k_coef + c_coef)
            step_intercept = lr * (g_intercept - c_k_intercept + c_intercept)
            local_coef = global_coef - step_coef
            local_intercept = global_intercept - step_intercept

            if x_design.shape[0] > 0:
                c_k_new_coef = c_k_coef - c_coef + (global_coef - local_coef) / lr
                c_k_new_intercept = (
                    c_k_intercept - c_intercept + (global_intercept - local_intercept) / lr
                )
                n_total += x_design.shape[0]
            else:
                c_k_new_coef = c_k_coef
                c_k_new_intercept = c_k_intercept

            new_coefs.append(local_coef)
            new_intercepts.append(local_intercept)
            weights.append(max(x_design.shape[0], 1))
            delta_c.append(
                (c_k_new_coef - c_k_coef, c_k_new_intercept - c_k_intercept)
            )
            client_c[cid] = (c_k_new_coef, c_k_new_intercept)

        w_arr = np.array(weights, dtype=float)
        global_coef = np.average(np.vstack(new_coefs), axis=0, weights=w_arr)
        global_intercept = float(np.average(new_intercepts, weights=w_arr))

        if delta_c:
            dc_coef = np.mean([d[0] for d in delta_c], axis=0)
            dc_intercept = float(np.mean([d[1] for d in delta_c]))
            c_coef = c_coef + dc_coef
            c_intercept = c_intercept + dc_intercept

    est = _fedavg_style_mean_prediction(global_coef, global_intercept, data)
    return EstimateResult(
        method="scaffold",
        estimate=est,
        n_moments=n_total,
    )


def estimate_fedpu_adapted(data: FederatedDataset, seed: int = 0) -> EstimateResult:
    """
    Federated positive-unlabeled proxy for E[Y(1)] under action-coded selection.

    Adaptation of Kiryo et al. (NeurIPS 2017) nnPU risk to scalar mean estimation:
    - action=1: labeled positive (observed outcome)
    - action=0: unlabeled (outcome hidden under selection)

    Per client k we estimate pi_k = P(A=1), fit mu_k(x) on observed positives,
    then form the nnPU-style moment
        pi_k * mean(y | A=1) + (1-pi_k)*mean(mu_k | A=0) - pi_k*mean(mu_k | A=1).
    Server aggregates moments with sample-size weights. This is an honest practical
    stub — not a full non-negative PU classifier — but debiases missing-label mass
    relative to naive positive-only means.
    """
    moment_sum = 0.0
    weight_sum = 0.0
    n_moments = 0

    for c in data.clients:
        n = c.x.shape[0]
        if n == 0:
            continue
        pi_k = float(np.mean(c.action == data.target_action))
        pi_k = float(np.clip(pi_k, 0.05, 0.95))

        pos_mask = c.observed
        unl_mask = ~c.observed

        if pos_mask.any():
            mu_model = _fit_outcome_regressor(c.x[pos_mask], c.y_obs[pos_mask])
            from .estimators import _predict_outcome

            mu_all = _predict_outcome(mu_model, c.x)
        else:
            mu_all = np.zeros(n)

        y_pos = c.y_obs[pos_mask]
        mean_y_pos = float(np.mean(y_pos)) if pos_mask.any() else 0.0
        mean_mu_unl = float(np.mean(mu_all[unl_mask])) if unl_mask.any() else 0.0
        mean_mu_pos = float(np.mean(mu_all[pos_mask])) if pos_mask.any() else 0.0

        pu_moment = pi_k * mean_y_pos + (1.0 - pi_k) * mean_mu_unl - pi_k * mean_mu_pos
        moment_sum += pu_moment * n
        weight_sum += n
        n_moments += n

    if weight_sum == 0:
        return EstimateResult(method="fedpu_adapted", estimate=float("nan"), n_moments=0)

    est = moment_sum / weight_sum
    if not np.isfinite(est):
        est = float("nan")

    return EstimateResult(
        method="fedpu_adapted",
        estimate=float(est),
        n_moments=n_moments,
    )
