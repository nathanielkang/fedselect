"""Client selection policies and propensity estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


def clip_propensity(p: np.ndarray, eps: float = 0.02) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _constant_propensity(action: np.ndarray, eps: float) -> float:
    rate = float(np.mean(action)) if action.size else 0.5
    return float(np.clip(rate, eps, 1.0 - eps))


@dataclass
class PropensityModel:
    """Cross-fitted propensity estimator for binary action."""

    eps: float = 0.02

    def fit_predict(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if len(np.unique(action)) < 2:
            const = _constant_propensity(action, self.eps)
            return np.full(x.shape[0], const)
        model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        model.fit(x, action)
        p = model.predict_proba(x)[:, 1]
        return clip_propensity(p, self.eps)

    def fit_predict_crossfit(
        self,
        x: np.ndarray,
        action: np.ndarray,
        seed: int = 0,
        n_folds: int = 2,
    ) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n = x.shape[0]
        if len(np.unique(action)) < 2:
            const = _constant_propensity(action, self.eps)
            return np.full(n, const)
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(n), n_folds)
        p_hat = np.zeros(n)
        for k in range(n_folds):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            y_train = action[train_idx]
            if len(np.unique(y_train)) < 2:
                const = _constant_propensity(y_train, self.eps)
                p_hat[test_idx] = const
                continue
            model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            model.fit(x[train_idx], y_train)
            p_hat[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        return clip_propensity(p_hat, self.eps)


def selection_indicator(action: np.ndarray, target_action: int = 1) -> np.ndarray:
    """Outcome observed iff action equals target (decision censoring)."""
    return action == target_action
