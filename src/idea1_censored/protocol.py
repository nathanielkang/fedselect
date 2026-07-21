"""Simulated federated rounds with secure moment aggregation (no raw data)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dgp import FederatedDataset
from .estimators import (
    EstimateResult,
    estimate_federated_dr,
    estimate_federated_ipw,
)
from .selection import PropensityModel


@dataclass
class RoundLog:
    round_id: int
    client_moment_sums: Dict[int, float]
    client_counts: Dict[int, int]
    aggregated_mean: float


@dataclass
class FederatedMomentProtocol:
    """
    Simulates T federated rounds where clients upload only scalar moment sums.
    """

    n_rounds: int = 1
    dp_eps: Optional[float] = None
    seed: int = 42
    logs: List[RoundLog] = field(default_factory=list)

    def _client_dr_moments(
        self, data: FederatedDataset, client_id: int
    ) -> Tuple[np.ndarray, PropensityModel]:
        prop_model = PropensityModel()
        c = data.clients[client_id]
        pi = prop_model.fit_predict_crossfit(c.x, c.action, seed=self.seed + client_id)
        mask = c.observed
        if mask.any():
            from .estimators import _fit_outcome_regressor, _predict_outcome

            reg = _fit_outcome_regressor(c.x[mask], c.y_obs[mask])
            mu_hat = _predict_outcome(reg, c.x)
        else:
            mu_hat = np.zeros(c.x.shape[0])
        a = c.action.astype(float)
        y = np.where(mask, c.y_obs, 0.0)
        treated = (a == data.target_action).astype(float)
        moments = mu_hat + treated * (y - mu_hat) / pi
        return moments, prop_model

    def run_dr_rounds(self, data: FederatedDataset) -> EstimateResult:
        self.logs.clear()
        all_sums = 0.0
        all_n = 0
        for r in range(self.n_rounds):
            round_sums: Dict[int, float] = {}
            round_counts: Dict[int, int] = {}
            for cid in range(len(data.clients)):
                moments, _ = self._client_dr_moments(data, cid)
                s = float(np.sum(moments))
                n = len(moments)
                if self.dp_eps is not None and self.dp_eps > 0:
                    scale = 5.0 / self.dp_eps
                    s += float(np.random.default_rng(self.seed + r + cid).normal(0.0, scale))
                round_sums[cid] = s
                round_counts[cid] = n
                all_sums += s
                all_n += n
            agg = all_sums / max(all_n, 1)
            self.logs.append(
                RoundLog(
                    round_id=r,
                    client_moment_sums=round_sums,
                    client_counts=round_counts,
                    aggregated_mean=agg,
                )
            )
        return EstimateResult(
            method="federated_dr_protocol",
            estimate=all_sums / max(all_n, 1),
            n_moments=all_n,
        )

    def run_ipw_rounds(self, data: FederatedDataset) -> EstimateResult:
        self.logs.clear()
        prop_model = PropensityModel()
        all_sums = 0.0
        all_n = 0
        for r in range(self.n_rounds):
            round_sums: Dict[int, float] = {}
            round_counts: Dict[int, int] = {}
            for cid, c in enumerate(data.clients):
                pi = prop_model.fit_predict_crossfit(c.x, c.action, seed=self.seed + cid)
                a = c.action.astype(float)
                y = np.where(c.observed, c.y_obs, 0.0)
                treated = (a == data.target_action).astype(float)
                moments = treated * y / pi
                s = float(np.sum(moments))
                n = len(moments)
                round_sums[cid] = s
                round_counts[cid] = n
                all_sums += s
                all_n += n
            self.logs.append(
                RoundLog(
                    round_id=r,
                    client_moment_sums=round_sums,
                    client_counts=round_counts,
                    aggregated_mean=all_sums / max(all_n, 1),
                )
            )
        return EstimateResult(
            method="federated_ipw_protocol",
            estimate=all_sums / max(all_n, 1),
            n_moments=all_n,
        )

    @staticmethod
    def one_shot_compare(data: FederatedDataset, seed: int = 42, dp_eps: Optional[float] = None) -> Dict[str, float]:
        """Convenience: protocol rounds vs closed-form estimators."""
        proto = FederatedMomentProtocol(n_rounds=2, dp_eps=dp_eps, seed=seed)
        dr_p = proto.run_dr_rounds(data)
        ipw_p = proto.run_ipw_rounds(data)
        dr = estimate_federated_dr(data, seed=seed, dp_eps=dp_eps)
        ipw = estimate_federated_ipw(data, seed=seed, dp_eps=dp_eps)
        return {
            "federated_dr_protocol": dr_p.estimate,
            "federated_ipw_protocol": ipw_p.estimate,
            "federated_dr": dr.estimate,
            "federated_ipw": ipw.estimate,
        }
