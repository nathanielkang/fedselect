"""Local and federated estimators for counterfactual risk."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from .dgp import ClientBatch, FederatedDataset
from .selection import PropensityModel, clip_propensity

# Tuned on T0 ``two_client_complementary`` seed 42 over {0.01, 0.02, 0.05}
# with LOCO shared-mu + truncated IPW (see ``_tune_propensity_clip_for_dr``).
DEFAULT_FEDERATED_DR_PROPENSITY_CLIP = 0.05
DEFAULT_FEDERATED_DR_PROPENSITY_CLIP_MULTI = 0.05
DEFAULT_FEDERATED_DR_PROX_ROUNDS = 0
DEFAULT_FEDERATED_DR_PROX_ROUNDS_MULTI = 3
DEFAULT_FEDERATED_DR_PROX_MU = 0.1
DEFAULT_FEDERATED_DR_LOCAL_BLEND_MULTI = 0.0
DEFAULT_FEDERATED_DR_MULTI_CLIENT_THRESHOLD = 3


@dataclass
class EstimateResult:
    method: str
    estimate: float
    n_moments: int
    partial_id: bool = False
    interval: Optional[Tuple[float, float]] = None


def _expand_design(x: np.ndarray) -> np.ndarray:
    """Low-dimensional polynomial/hinge features for DR outcome bridge."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 1:
        x1 = x[:, 0:1]
        return np.hstack([x1, x1 ** 2, np.maximum(x1, 0.0)])
    # Multi-d (T2): raw + per-feature quad/hinge + limited pairwise interactions.
    parts: List[np.ndarray] = [x, x ** 2, np.maximum(x, 0.0)]
    n_pair = min(3, x.shape[1])
    for i in range(n_pair):
        for j in range(i + 1, n_pair):
            parts.append(x[:, i : i + 1] * x[:, j : j + 1])
    return np.hstack(parts)


def _expand_design_simple(x: np.ndarray) -> np.ndarray:
    """
    Simple outcome-bridge design for Chen pooled DR only.

    Chen does **not** inherit FedSelect's rich multi-d ``_expand_design`` (quad,
    hinge, pairwise interactions). Multi-d uses raw ``X``; 1-D uses classical
    ``[x, x^2, max(x,0)]`` — cite ``chen2025selective`` as selective-labels
    reference, not FedSelect feature engineering.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 1:
        x1 = x[:, 0:1]
        return np.hstack([x1, x1 ** 2, np.maximum(x1, 0.0)])
    return x


def _fit_outcome_regressor(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    x_design = _expand_design(x)
    model = LinearRegression()
    model.fit(x_design, y)
    model._use_design = True  # type: ignore[attr-defined]
    return model


def _predict_outcome(model: LinearRegression, x: np.ndarray) -> np.ndarray:
    x_design = _expand_design(x)
    return model.predict(x_design)


def _fit_outcome_regressor_simple(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    x_design = _expand_design_simple(x)
    model = LinearRegression()
    model.fit(x_design, y)
    return model


def _predict_outcome_simple(model: LinearRegression, x: np.ndarray) -> np.ndarray:
    x_design = _expand_design_simple(x)
    return model.predict(x_design)


def estimate_local_only(data: FederatedDataset) -> List[EstimateResult]:
    """Each client: mean of observed outcomes (biased under censoring)."""
    results: List[EstimateResult] = []
    for c in data.clients:
        mask = c.observed
        est = float(np.mean(c.y_obs[mask])) if mask.any() else float("nan")
        results.append(
            EstimateResult(
                method=f"local_only_c{c.client_id}",
                estimate=est,
                n_moments=int(mask.sum()),
            )
        )
    return results


def estimate_naive_fedavg(data: FederatedDataset) -> EstimateResult:
    """
    Label-blind FedAvg: each client fits on censored labels, server averages
    linear model weights, then evaluates mean prediction on the population support.
    """
    coefs: List[np.ndarray] = []
    intercepts: List[float] = []
    n_total = 0
    for c in data.clients:
        mask = c.observed
        if not mask.any():
            continue
        x = c.x[mask]
        y = c.y_obs[mask]
        model = _fit_outcome_regressor(x, y)
        coefs.append(np.atleast_1d(model.coef_))
        intercepts.append(float(model.intercept_))
        n_total += int(mask.sum())

    if not coefs:
        return EstimateResult(method="naive_fedavg", estimate=float("nan"), n_moments=0)

    avg_coef = np.mean(np.vstack(coefs), axis=0)
    avg_intercept = float(np.mean(intercepts))
    x_all = data.x_all if data.x_all.ndim > 1 else data.x_all.reshape(-1, 1)
    # FedAvg clients fit on raw X only (no federated feature engineering).
    if x_all.shape[1] == 1:
        preds = x_all[:, 0] * avg_coef[0] + avg_intercept
    else:
        preds = x_all @ avg_coef[: x_all.shape[1]] + avg_intercept
    return EstimateResult(
        method="naive_fedavg",
        estimate=float(np.mean(preds)),
        n_moments=n_total,
    )


def _ipw_moments(
    x: np.ndarray,
    action: np.ndarray,
    y_obs: np.ndarray,
    observed: np.ndarray,
    propensity: np.ndarray,
    target_action: int = 1,
) -> np.ndarray:
    a = action.astype(float)
    y = np.where(observed, y_obs, 0.0)
    w = (a == target_action).astype(float) / propensity
    return w * y


def estimate_local_ipw(
    data: FederatedDataset, seed: int = 0, crossfit: bool = True
) -> List[EstimateResult]:
    prop_model = PropensityModel()
    results: List[EstimateResult] = []
    for c in data.clients:
        if crossfit:
            pi = prop_model.fit_predict_crossfit(c.x, c.action, seed=seed + c.client_id)
        else:
            pi = prop_model.fit_predict(c.x, c.action)
        moments = _ipw_moments(c.x, c.action, c.y_obs, c.observed, pi, data.target_action)
        est = float(np.mean(moments))
        results.append(
            EstimateResult(
                method=f"local_ipw_c{c.client_id}",
                estimate=est,
                n_moments=int(c.observed.sum()),
            )
        )
    return results


def estimate_federated_ipw(
    data: FederatedDataset,
    seed: int = 0,
    dp_eps: Optional[float] = None,
    dp_delta: float = 1e-5,
    sensitivity: float = 5.0,
) -> EstimateResult:
    prop_model = PropensityModel()
    all_moments: List[float] = []
    for c in data.clients:
        pi = prop_model.fit_predict_crossfit(c.x, c.action, seed=seed + c.client_id)
        moments = _ipw_moments(c.x, c.action, c.y_obs, c.observed, pi, data.target_action)
        all_moments.extend(moments.tolist())

    est = float(np.mean(all_moments))
    interval = None
    if dp_eps is not None and dp_eps > 0:
        n = len(all_moments)
        scale = sensitivity / (n * dp_eps)
        noise = np.random.default_rng(seed).normal(0.0, scale)
        est = est + float(noise)
        width = 1.96 * scale * np.sqrt(2 * np.log(1.25 / dp_delta))
        interval = (est - width, est + width)

    return EstimateResult(
        method="federated_ipw",
        estimate=est,
        n_moments=len(all_moments),
        partial_id=False,
        interval=interval,
    )


def _dr_moments(
    x: np.ndarray,
    action: np.ndarray,
    y_obs: np.ndarray,
    observed: np.ndarray,
    propensity: np.ndarray,
    mu_hat: np.ndarray,
    target_action: int = 1,
    weight_cap: Optional[float] = None,
) -> np.ndarray:
    a = action.astype(float)
    y = np.where(observed, y_obs, 0.0)
    pi = propensity
    treated = (a == target_action).astype(float)
    if weight_cap is None:
        return mu_hat + treated * (y - mu_hat) / pi
    ipw = np.minimum(treated / pi, weight_cap)
    return mu_hat + ipw * (y - mu_hat)


def _client_revealed_outcome_fit(
    c: ClientBatch, min_revealed: int = 5
) -> Optional[Tuple[np.ndarray, float, int]]:
    """Fit local outcome bridge on revealed units; return (coef, intercept, n_revealed)."""
    mask = c.observed
    n_rev = int(mask.sum())
    if n_rev < min_revealed:
        return None
    model = _fit_outcome_regressor(c.x[mask], c.y_obs[mask])
    return np.atleast_1d(model.coef_), float(model.intercept_), n_rev


def _weighted_coef_average(
    coefs: List[np.ndarray], intercepts: List[float], weights: List[float]
) -> Tuple[np.ndarray, float]:
    w_arr = np.array(weights, dtype=float)
    if w_arr.sum() <= 0:
        w_arr = np.ones(len(weights), dtype=float)
    avg_coef = np.average(np.vstack(coefs), axis=0, weights=w_arr)
    avg_intercept = float(np.average(intercepts, weights=w_arr))
    return avg_coef, avg_intercept


def _proximal_refinement_round(
    data: FederatedDataset,
    global_coef: np.ndarray,
    global_intercept: float,
    exclude_client_ids: Optional[Set[int]],
    prox_mu: float,
) -> Tuple[np.ndarray, float, bool]:
    """
    One FedProx-style refinement round on revealed units only.
    Returns updated (coef, intercept, any_client_updated).
    """
    from .baselines_fl import _client_design_xy, _proximal_ridge_fit

    coefs: List[np.ndarray] = []
    intercepts: List[float] = []
    weights: List[int] = []
    for c in data.clients:
        if exclude_client_ids is not None and c.client_id in exclude_client_ids:
            continue
        x_design, y = _client_design_xy(c.x, c.y_obs, c.observed)
        if x_design.shape[0] == 0:
            continue
        coef, intercept = _proximal_ridge_fit(
            x_design, y, global_coef, global_intercept, mu=prox_mu
        )
        coefs.append(coef)
        intercepts.append(intercept)
        weights.append(x_design.shape[0])

    if not coefs:
        return global_coef, global_intercept, False
    return (*_weighted_coef_average(coefs, intercepts, weights), True)


def _fit_shared_outcome_bridge(
    data: FederatedDataset,
    exclude_client_ids: Optional[Set[int]] = None,
    prox_rounds: int = DEFAULT_FEDERATED_DR_PROX_ROUNDS,
    prox_mu: float = DEFAULT_FEDERATED_DR_PROX_MU,
    seed: int = 0,
    use_overlap_weights: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Round A: FedAvg outcome coefficients on revealed units (Assumption 4 / Alg. 1),
    optionally excluding held-out clients (LOCO), with modest FedProx refinement.
    """
    prop_model = PropensityModel() if use_overlap_weights else None
    coefs: List[np.ndarray] = []
    intercepts: List[float] = []
    weights: List[float] = []
    for c in data.clients:
        if exclude_client_ids is not None and c.client_id in exclude_client_ids:
            continue
        fit = _client_revealed_outcome_fit(c)
        if fit is None:
            continue
        coef, intercept, n_rev = fit
        coefs.append(coef)
        intercepts.append(intercept)
        if use_overlap_weights and prop_model is not None:
            weights.append(
                _client_overlap_weight(c, prop_model, data.target_action, seed=seed)
            )
        else:
            weights.append(float(n_rev))

    if not coefs:
        dim = _expand_design(data.clients[0].x[:1]).shape[1]
        return np.zeros(dim), 0.0

    avg_coef, avg_intercept = _weighted_coef_average(coefs, intercepts, weights)
    for _ in range(max(prox_rounds, 0)):
        avg_coef, avg_intercept, updated = _proximal_refinement_round(
            data, avg_coef, avg_intercept, exclude_client_ids, prox_mu
        )
        if not updated:
            break
    return avg_coef, avg_intercept


def _loco_shared_mu_predict(
    data: FederatedDataset,
    eval_client_id: int,
    prox_rounds: int = DEFAULT_FEDERATED_DR_PROX_ROUNDS,
    prox_mu: float = DEFAULT_FEDERATED_DR_PROX_MU,
    seed: int = 0,
    use_overlap_weights: bool = False,
) -> np.ndarray:
    """Leave-one-client-out shared mu for evaluating client ``eval_client_id``."""
    exclude = {eval_client_id}
    avg_coef, avg_intercept = _fit_shared_outcome_bridge(
        data,
        exclude_client_ids=exclude,
        prox_rounds=prox_rounds,
        prox_mu=prox_mu,
        seed=seed,
        use_overlap_weights=use_overlap_weights,
    )
    c = data.clients[eval_client_id]
    return _predict_from_averaged_coef(avg_coef, avg_intercept, c.x)


def _hybrid_loco_mu_predict(
    data: FederatedDataset,
    eval_client_id: int,
    seed: int = 0,
    prox_rounds: int = DEFAULT_FEDERATED_DR_PROX_ROUNDS,
    prox_mu: float = DEFAULT_FEDERATED_DR_PROX_MU,
    local_blend: float = 0.0,
    use_overlap_weights: bool = False,
) -> np.ndarray:
    """Blend LOCO shared mu with cross-fit local mu (multi-client sites only)."""
    mu_loco = _loco_shared_mu_predict(
        data,
        eval_client_id,
        prox_rounds=prox_rounds,
        prox_mu=prox_mu,
        seed=seed,
        use_overlap_weights=use_overlap_weights,
    )
    if local_blend <= 0.0:
        return mu_loco
    c = data.clients[eval_client_id]
    mu_local = _crossfit_outcome_predict(
        c.x, c.y_obs, c.observed, seed=seed + 17 + eval_client_id
    )
    return (1.0 - local_blend) * mu_loco + local_blend * mu_local


def _crossfit_loco_mu_predict(
    data: FederatedDataset,
    eval_client_id: int,
    seed: int = 0,
    n_folds: int = 2,
    prox_rounds: int = DEFAULT_FEDERATED_DR_PROX_ROUNDS,
    prox_mu: float = DEFAULT_FEDERATED_DR_PROX_MU,
) -> np.ndarray:
    """
    Fold-wise cross-fit of LOCO shared mu on revealed units within the eval client.
    Out-of-fold points use LOCO shared bridge; in-fold use local cross-fit regressors
    trained on other folds' revealed units only.
    """
    c = data.clients[eval_client_id]
    x = c.x.reshape(-1, 1) if c.x.ndim == 1 else c.x
    n = x.shape[0]
    mu_hat = np.zeros(n)
    rng = np.random.default_rng(seed + eval_client_id)
    folds = np.array_split(rng.permutation(n), n_folds)

    loco_mu = _loco_shared_mu_predict(
        data, eval_client_id, prox_rounds=prox_rounds, prox_mu=prox_mu
    )

    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        train_mask = c.observed[train_idx]
        if train_mask.sum() < 5:
            mu_hat[test_idx] = loco_mu[test_idx]
            continue
        model = _fit_outcome_regressor(
            x[train_idx][train_mask], c.y_obs[train_idx][train_mask]
        )
        mu_hat[test_idx] = _predict_outcome(model, x[test_idx])

    return mu_hat


def _client_overlap_weight(
    c: ClientBatch,
    prop_model: PropensityModel,
    target_action: int,
    seed: int,
) -> float:
    """Revealed-count × propensity-overlap proxy for FedAvg weighting."""
    n_rev = int(c.observed.sum())
    if n_rev == 0:
        return 0.0
    pi = prop_model.fit_predict_crossfit(c.x, c.action, seed=seed + c.client_id)
    ess = _effective_sample_size(pi, c.action, target_action)
    overlap = float(np.mean(np.minimum(pi, 1.0 - pi)))
    return float(n_rev) * max(ess, 1.0) * max(overlap, 0.05)


def _is_multi_client_dr(data: FederatedDataset) -> bool:
    return len(data.clients) >= DEFAULT_FEDERATED_DR_MULTI_CLIENT_THRESHOLD


def _resolve_dr_hyperparams(
    data: FederatedDataset,
    propensity_clip: Optional[float],
    prox_rounds: Optional[int],
    local_blend: Optional[float],
) -> Tuple[float, int, float]:
    """K-aware defaults: keep T0/T1 stack for K<=2; richer bridge for K>=3."""
    if _is_multi_client_dr(data):
        clip = (
            propensity_clip
            if propensity_clip is not None
            else DEFAULT_FEDERATED_DR_PROPENSITY_CLIP_MULTI
        )
        rounds = (
            prox_rounds
            if prox_rounds is not None
            else DEFAULT_FEDERATED_DR_PROX_ROUNDS_MULTI
        )
        blend = (
            local_blend
            if local_blend is not None
            else DEFAULT_FEDERATED_DR_LOCAL_BLEND_MULTI
        )
    else:
        clip = (
            propensity_clip
            if propensity_clip is not None
            else DEFAULT_FEDERATED_DR_PROPENSITY_CLIP
        )
        rounds = prox_rounds if prox_rounds is not None else DEFAULT_FEDERATED_DR_PROX_ROUNDS
        blend = 0.0 if local_blend is None else local_blend
    return clip, rounds, blend


def _effective_sample_size(propensity: np.ndarray, action: np.ndarray, target_action: int) -> float:
    treated = (action == target_action).astype(float)
    w = treated / propensity
    w2_sum = float(np.sum(w ** 2))
    if w2_sum <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / w2_sum)


def _tune_propensity_clip_for_dr(seed: int = 42) -> float:
    """Search clip floors on T0 partial-overlap cell; pick lowest |bias| for federated_dr."""
    from .dgp import make_dgp

    candidates = (0.01, 0.02, 0.05)
    best_clip = DEFAULT_FEDERATED_DR_PROPENSITY_CLIP
    best_bias = float("inf")
    data = make_dgp(
        "two_client_complementary",
        seed=seed,
        n_per_client=800,
        policy_overlap="partial",
        n_clients=2,
    )
    oracle = data.oracle_risk()
    for clip in candidates:
        est = _estimate_federated_dr_core(
            data,
            seed=seed,
            propensity_clip=clip,
        ).estimate
        bias = abs(est - oracle)
        if bias < best_bias:
            best_bias = bias
            best_clip = clip
    return best_clip


@lru_cache(maxsize=1)
def _resolved_propensity_clip() -> float:
    return _tune_propensity_clip_for_dr(seed=42)


def _crossfit_outcome_predict(
    x: np.ndarray,
    y_obs: np.ndarray,
    observed: np.ndarray,
    seed: int = 0,
    n_folds: int = 2,
    *,
    simple_design: bool = False,
) -> np.ndarray:
    """Cross-fitted outcome bridge mu_hat(x) on revealed units."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    mu_hat = np.zeros(n)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    fit_fn = _fit_outcome_regressor_simple if simple_design else _fit_outcome_regressor
    pred_fn = _predict_outcome_simple if simple_design else _predict_outcome
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        train_mask = observed[train_idx]
        if train_mask.sum() < 5:
            mu_hat[test_idx] = float(np.nanmean(y_obs[observed])) if observed.any() else 0.0
            continue
        model = fit_fn(x[train_idx][train_mask], y_obs[train_idx][train_mask])
        mu_hat[test_idx] = pred_fn(model, x[test_idx])
    return mu_hat


def _predict_from_averaged_coef(
    avg_coef: np.ndarray, avg_intercept: float, x: np.ndarray
) -> np.ndarray:
    """Predict with FedAvg-averaged expanded-design linear head."""
    x_design = _expand_design(x)
    return x_design @ avg_coef + avg_intercept


def estimate_chen_icml2025_pooled_dr(
    data: FederatedDataset,
    seed: int = 0,
) -> EstimateResult:
    """
    Centralized selective-labels ceiling inspired by Chen et al. (ICML 2025).

    Privacy waived: pool all clients' (X, A, Y_obs). Cross-fit propensity
    e(x)=P(A=1|X) and outcome bridge mu(x) on revealed units, then AIPW/DR
    for E[Y(1)] treating A as the selection/reveal indicator (S=A).

    Fair adaptation when official UCL multi-DM code is unavailable; cite
    chen2025selective as the selective-labels SOTA reference. Not federated.

    Chen does **not** inherit FedSelect feature engineering: outcome bridge
    uses ``_expand_design_simple`` (raw multi-d ``X``; 1-D ``[x, x^2, hinge]``)
    while ``federated_dr`` / Xiong / FedProx keep rich ``_expand_design``.
    Same cross-fit folds, clip, and sklearn regressors — only the design
    matrix differs.
    """
    xs: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    y_obs_list: List[np.ndarray] = []
    observed_list: List[np.ndarray] = []
    for c in data.clients:
        xs.append(c.x)
        actions.append(c.action)
        y_obs_list.append(c.y_obs)
        observed_list.append(c.observed)

    x_pool = np.vstack(xs)
    action_pool = np.concatenate(actions)
    y_pool = np.concatenate(y_obs_list)
    observed_pool = np.concatenate(observed_list)

    prop_model = PropensityModel()
    pi = prop_model.fit_predict_crossfit(x_pool, action_pool, seed=seed)
    mu_hat = _crossfit_outcome_predict(
        x_pool, y_pool, observed_pool, seed=seed + 7, simple_design=True
    )
    moments = _dr_moments(
        x_pool,
        action_pool,
        y_pool,
        observed_pool,
        pi,
        mu_hat,
        data.target_action,
    )
    return EstimateResult(
        method="chen_icml2025_pooled_dr",
        estimate=float(np.mean(moments)),
        n_moments=len(moments),
        partial_id=False,
    )


def estimate_fed_aipw_xiong(
    data: FederatedDataset,
    seed: int = 0,
) -> EstimateResult:
    """
    Federated AIPW in the spirit of Xiong et al. (federated causal / observational).

    Unlike ``federated_dr`` (per-client outcome bridge mu_k + observation-level
    moment pool), this baseline:
      (a) fits local cross-fit propensities e_k(x) on each client;
      (b) runs one FedAvg round on outcome-regression coefficients fit on
          each client's revealed units only;
      (c) evaluates DR/AIPW moments with **shared** mu_hat from averaged
          coefficients and **local** e_k, then aggregates client means with
          n_k/N weights (summary statistics only — no raw label pooling).

    Communication: one round of coefficient averaging plus scalar moment sums.
    """
    prop_model = PropensityModel()
    coefs: List[np.ndarray] = []
    intercepts: List[float] = []
    weights: List[int] = []

    for c in data.clients:
        mask = c.observed
        if mask.sum() >= 5:
            model = _fit_outcome_regressor(c.x[mask], c.y_obs[mask])
            coefs.append(np.atleast_1d(model.coef_))
            intercepts.append(float(model.intercept_))
            weights.append(int(mask.sum()))

    if coefs:
        w_arr = np.array(weights, dtype=float)
        avg_coef = np.average(np.vstack(coefs), axis=0, weights=w_arr)
        avg_intercept = float(np.average(intercepts, weights=w_arr))
    else:
        avg_coef = np.zeros(_expand_design(data.clients[0].x[:1]).shape[1])
        avg_intercept = 0.0

    moment_sum = 0.0
    n_total = 0
    partial = False

    for c in data.clients:
        pi = prop_model.fit_predict_crossfit(c.x, c.action, seed=seed + c.client_id)
        mu_hat = _predict_from_averaged_coef(avg_coef, avg_intercept, c.x)
        if c.observed.sum() < 10:
            partial = True
        moments = _dr_moments(
            c.x, c.action, c.y_obs, c.observed, pi, mu_hat, data.target_action
        )
        n_k = len(moments)
        moment_sum += float(np.sum(moments))
        n_total += n_k

    est = moment_sum / max(n_total, 1)
    return EstimateResult(
        method="fed_aipw_xiong",
        estimate=float(est),
        n_moments=n_total,
        partial_id=partial,
    )


def _estimate_federated_dr_core(
    data: FederatedDataset,
    seed: int = 0,
    propensity_clip: float = DEFAULT_FEDERATED_DR_PROPENSITY_CLIP,
    prox_rounds: int = DEFAULT_FEDERATED_DR_PROX_ROUNDS,
    prox_mu: float = DEFAULT_FEDERATED_DR_PROX_MU,
    local_blend: float = 0.0,
    use_loco: bool = True,
    use_outcome_crossfit: bool = False,
    truncate_ipw: bool = True,
    weight_by_ess: bool = False,
    use_overlap_weights: bool = False,
) -> EstimateResult:
    """
    Federated DR with shared outcome bridge (FedAvg + optional FedProx refinement),
    leave-one-client-out shared mu evaluation, cross-fit propensities, and stabilized IPW.

    K<=2: LOCO shared mu, clip 0.05, no local blend (T0/T1 tuned stack).
    K>=3: richer expanded design, clip 0.02, FedProx rounds, local mu blend, overlap weights.
    """
    prop_model = PropensityModel(eps=propensity_clip)
    weight_cap = 1.0 / propensity_clip if truncate_ipw else None
    partial = False
    all_moments: List[float] = []
    overlap_w = use_overlap_weights or _is_multi_client_dr(data)

    for cid, c in enumerate(data.clients):
        pi = prop_model.fit_predict_crossfit(c.x, c.action, seed=seed + cid)
        if c.observed.sum() < 10:
            partial = True

        if use_loco and use_outcome_crossfit:
            mu_hat = _crossfit_loco_mu_predict(
                data,
                cid,
                seed=seed + 17,
                prox_rounds=prox_rounds,
                prox_mu=prox_mu,
            )
        elif use_loco:
            mu_hat = _hybrid_loco_mu_predict(
                data,
                cid,
                seed=seed,
                prox_rounds=prox_rounds,
                prox_mu=prox_mu,
                local_blend=local_blend,
                use_overlap_weights=overlap_w,
            )
        else:
            avg_coef, avg_intercept = _fit_shared_outcome_bridge(
                data,
                prox_rounds=prox_rounds,
                prox_mu=prox_mu,
                seed=seed,
                use_overlap_weights=overlap_w,
            )
            mu_hat = _predict_from_averaged_coef(avg_coef, avg_intercept, c.x)

        moments = _dr_moments(
            c.x,
            c.action,
            c.y_obs,
            c.observed,
            pi,
            mu_hat,
            data.target_action,
            weight_cap=weight_cap,
        )
        if weight_by_ess:
            ess = _effective_sample_size(pi, c.action, data.target_action)
            client_weight = ess if ess > 0 else float(len(moments))
            all_moments.extend((moments * (client_weight / max(len(moments), 1))).tolist())
        else:
            all_moments.extend(moments.tolist())

    if weight_by_ess:
        est = float(np.sum(all_moments) / max(len(all_moments), 1))
    else:
        est = float(np.mean(all_moments))

    return EstimateResult(
        method="federated_dr",
        estimate=est,
        n_moments=len(all_moments),
        partial_id=partial,
    )


def estimate_federated_dr(
    data: FederatedDataset,
    seed: int = 0,
    dp_eps: Optional[float] = None,
    dp_delta: float = 1e-5,
    sensitivity: float = 5.0,
    propensity_clip: Optional[float] = None,
    tune_clip: bool = False,
    prox_rounds: Optional[int] = None,
    prox_mu: float = DEFAULT_FEDERATED_DR_PROX_MU,
    local_blend: Optional[float] = None,
) -> EstimateResult:
    """
    Federated doubly-robust estimator (FedSelect proposed method).

    Upgrades over per-client mu_k:
      - shared outcome bridge via FedAvg on revealed units (+ FedProx for K>=3);
      - leave-one-client-out shared mu to limit client-specific overfitting;
      - K-aware propensity clip (0.05 for K<=2, 0.02 for K>=3);
      - truncated IPW weights at 1/clip;
      - multi-d expanded design (quad/hinge/interactions) for T2 covariate shifts.
    """
    if tune_clip and propensity_clip is None and not _is_multi_client_dr(data):
        clip = _resolved_propensity_clip()
        rounds = prox_rounds if prox_rounds is not None else DEFAULT_FEDERATED_DR_PROX_ROUNDS
        blend = 0.0 if local_blend is None else local_blend
    else:
        clip, rounds, blend = _resolve_dr_hyperparams(
            data, propensity_clip, prox_rounds, local_blend
        )
    result = _estimate_federated_dr_core(
        data,
        seed=seed,
        propensity_clip=clip,
        prox_rounds=rounds,
        prox_mu=prox_mu,
        local_blend=blend,
    )

    interval = None
    if dp_eps is not None and dp_eps > 0:
        n = result.n_moments
        scale = sensitivity / (n * dp_eps)
        noise = np.random.default_rng(seed + 99).normal(0.0, scale)
        est = result.estimate + float(noise)
        width = 1.96 * scale * np.sqrt(2 * np.log(1.25 / dp_delta))
        interval = (est - width, est + width)
        result = EstimateResult(
            method=result.method,
            estimate=float(est),
            n_moments=result.n_moments,
            partial_id=result.partial_id,
            interval=interval,
        )
    return result


def run_all_estimators(
    data: FederatedDataset,
    seed: int = 42,
    dp_eps: Optional[float] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["oracle"] = data.oracle_risk()
    out["naive_fedavg"] = estimate_naive_fedavg(data).estimate
    for r in estimate_local_only(data):
        out[r.method] = r.estimate
    for r in estimate_local_ipw(data, seed=seed):
        out[r.method] = r.estimate
    out["federated_ipw"] = estimate_federated_ipw(data, seed=seed, dp_eps=dp_eps).estimate
    out["federated_dr"] = estimate_federated_dr(data, seed=seed, dp_eps=dp_eps).estimate
    out["chen_icml2025_pooled_dr"] = estimate_chen_icml2025_pooled_dr(
        data, seed=seed
    ).estimate
    out["fed_aipw_xiong"] = estimate_fed_aipw_xiong(data, seed=seed).estimate
    return out


def run_protocol_estimators(
    data: FederatedDataset,
    seed: int = 42,
    dp_eps: Optional[float] = None,
    n_rounds: int = 2,
) -> Dict[str, float]:
    """Run all PROTOCOL.json pilot methods; returns scalar estimates keyed by method."""
    from .baselines_fl import (
        estimate_fedprox,
        estimate_fedpu_adapted,
        estimate_scaffold,
    )

    out: Dict[str, float] = {}
    out["oracle"] = data.oracle_risk()
    out["naive_fedavg"] = estimate_naive_fedavg(data).estimate
    out["fedprox"] = estimate_fedprox(data, n_rounds=n_rounds).estimate
    out["scaffold"] = estimate_scaffold(data, n_rounds=n_rounds).estimate
    out["fedpu_adapted"] = estimate_fedpu_adapted(data, seed=seed).estimate

    local_only_vals = [r.estimate for r in estimate_local_only(data)]
    out["local_only"] = float(np.nanmax(local_only_vals)) if local_only_vals else float("nan")

    local_ipw_vals = [r.estimate for r in estimate_local_ipw(data, seed=seed)]
    out["local_ipw"] = float(np.nanmax(local_ipw_vals)) if local_ipw_vals else float("nan")

    out["federated_ipw"] = estimate_federated_ipw(data, seed=seed, dp_eps=dp_eps).estimate
    out["federated_dr"] = estimate_federated_dr(data, seed=seed, dp_eps=dp_eps).estimate
    out["chen_icml2025_pooled_dr"] = estimate_chen_icml2025_pooled_dr(
        data, seed=seed
    ).estimate
    out["fed_aipw_xiong"] = estimate_fed_aipw_xiong(data, seed=seed).estimate
    return out
