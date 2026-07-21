"""Synthetic and semi-synthetic DGPs with known counterfactual outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml

_FOLKTABLES_AVAILABLE = False
try:
    from folktables import ACSIncome  # type: ignore

    _FOLKTABLES_AVAILABLE = True
except ImportError:
    ACSIncome = None  # type: ignore[misc, assignment]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _overlap_policy_logits(
    client_id: int,
    n_clients: int,
    x: np.ndarray,
    overlap: str,
    partial_logits: np.ndarray,
    strength: float = 4.0,
) -> np.ndarray:
    """Map stress-axis policy_overlap to client propensity logits."""
    x2 = x if x.ndim > 1 else x.reshape(-1, 1)
    if overlap == "full":
        pooled = np.mean(x2[:, : min(3, x2.shape[1])], axis=1)
        jitter = 0.2 * np.sin(2.0 * np.pi * client_id / max(n_clients, 1))
        return strength * 0.55 * pooled + jitter
    if overlap == "none":
        quantile = (client_id + 0.5) / n_clients
        threshold = np.quantile(x2[:, 0], quantile)
        return strength * 3.0 * (x2[:, 0] - threshold)
    return partial_logits


@dataclass
class ClientBatch:
    """One client's policy-censored observations."""

    client_id: int
    x: np.ndarray
    action: np.ndarray
    y_obs: np.ndarray
    observed: np.ndarray
    propensity: np.ndarray
    y1: np.ndarray
    y0: np.ndarray


@dataclass
class FederatedDataset:
    """Full federated draw with oracle counterfactuals."""

    clients: List[ClientBatch]
    x_all: np.ndarray
    y1_all: np.ndarray
    y0_all: np.ndarray
    target_action: int = 1

    def oracle_risk(self, action: Optional[int] = None) -> float:
        a = self.target_action if action is None else action
        y = self.y1_all if a == 1 else self.y0_all
        return float(np.mean(y))


class TwoClientComplementaryDGP:
    """
    Two-client synthetic DGP where complementary action policies create
    partial overlap: each client acts on opposite sides of X.
    """

    def __init__(
        self,
        n_per_client: int = 2000,
        intercept: float = 2.5,
        beta_x: float = 1.2,
        beta_x2: float = 0.8,
        noise: float = 0.4,
        policy_strength: float = 4.5,
        seed: int = 42,
        policy_overlap: str = "partial",
        n_clients: int = 2,
    ) -> None:
        self.n_per_client = n_per_client
        self.intercept = intercept
        self.beta_x = beta_x
        self.beta_x2 = beta_x2
        self.noise = noise
        self.policy_strength = policy_strength
        self.seed = seed
        self.policy_overlap = policy_overlap
        self.n_clients = n_clients

    def _client_policy(self, client_id: int, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0] if x.ndim > 1 else x
        sign = 1.0 if client_id % 2 == 0 else -1.0
        partial_logits = sign * self.policy_strength * x1
        x2 = x if x.ndim > 1 else x.reshape(-1, 1)
        logits = _overlap_policy_logits(
            client_id,
            self.n_clients,
            x2,
            self.policy_overlap,
            partial_logits,
            strength=self.policy_strength,
        )
        return _sigmoid(logits)

    def sample(self) -> FederatedDataset:
        rng = np.random.default_rng(self.seed)
        clients: List[ClientBatch] = []
        xs: List[np.ndarray] = []
        y1s: List[np.ndarray] = []
        y0s: List[np.ndarray] = []

        for cid in range(self.n_clients):
            x = rng.normal(0.0, 1.0, size=self.n_per_client)
            # Asymmetric surface: treated-high-X regions carry extra risk mass.
            y1 = (
                self.intercept
                + self.beta_x * x
                + 1.1 * np.maximum(x, 0.0)
                + rng.normal(0.0, self.noise, size=self.n_per_client)
            )
            y0 = 0.5 * x + rng.normal(0.0, self.noise, size=self.n_per_client)
            pi = self._client_policy(cid, x)
            action = rng.binomial(1, pi).astype(np.int8)
            observed = action == 1
            y_obs = np.where(observed, y1, np.nan)

            clients.append(
                ClientBatch(
                    client_id=cid,
                    x=x.reshape(-1, 1),
                    action=action,
                    y_obs=y_obs,
                    observed=observed,
                    propensity=pi,
                    y1=y1,
                    y0=y0,
                )
            )
            xs.append(x)
            y1s.append(y1)
            y0s.append(y0)

        return FederatedDataset(
            clients=clients,
            x_all=np.concatenate(xs),
            y1_all=np.concatenate(y1s),
            y0_all=np.concatenate(y0s),
            target_action=1,
        )


class SemiSyntheticCreditDGP:
    """
    Adult-style semi-synthetic credit risk with site-specific approval policies.

    Uses OpenML Adult (income) features; synthetic default outcome Y(1), Y(0)
    with known counterfactuals for evaluation.
    """

    def __init__(
        self,
        n_per_client: int = 1500,
        seed: int = 42,
        policy_strength: float = 2.2,
        policy_overlap: str = "partial",
        n_clients: int = 2,
    ) -> None:
        self.n_per_client = n_per_client
        self.seed = seed
        self.policy_strength = policy_strength
        self.policy_overlap = policy_overlap
        self.n_clients = n_clients

    def _load_features(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        try:
            data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
            df = data.frame
            y_raw = (df["class"] == ">50K").astype(float).to_numpy()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            x_df = df[num_cols].fillna(0.0)
            x = x_df.to_numpy(dtype=float)
            if x.shape[0] > 5000:
                idx = rng.choice(x.shape[0], size=5000, replace=False)
                x = x[idx]
                y_raw = y_raw[idx]
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
            return x, y_raw
        except Exception:
            rng_fallback = np.random.default_rng(self.seed)
            x = rng_fallback.normal(size=(4000, 6))
            y_raw = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(float)
            return x, y_raw

    def _outcome_models(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        score = 0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2]
        y1 = 1.5 + score + 0.35 * (score ** 2) + rng.normal(0.0, 0.5, size=x.shape[0])
        y0 = 0.3 * score + rng.normal(0.0, 0.5, size=x.shape[0])
        return y1, y0

    def _client_policy(self, client_id: int, x: np.ndarray) -> np.ndarray:
        if client_id % 2 == 0:
            partial = self.policy_strength * (x[:, 0] - 0.3 * x[:, 1])
        else:
            partial = self.policy_strength * (-x[:, 0] + 0.2 * x[:, min(2, x.shape[1] - 1)])
        logits = _overlap_policy_logits(
            client_id,
            self.n_clients,
            x,
            self.policy_overlap,
            partial,
            strength=self.policy_strength,
        )
        return _sigmoid(logits)

    def sample(self) -> FederatedDataset:
        rng = np.random.default_rng(self.seed)
        x_pool, _ = self._load_features(rng)
        y1_pool, y0_pool = self._outcome_models(x_pool, rng)

        clients: List[ClientBatch] = []
        xs: List[np.ndarray] = []
        y1s: List[np.ndarray] = []
        y0s: List[np.ndarray] = []

        for cid in range(self.n_clients):
            idx = rng.choice(x_pool.shape[0], size=self.n_per_client, replace=True)
            x = x_pool[idx]
            y1 = y1_pool[idx]
            y0 = y0_pool[idx]
            pi = self._client_policy(cid, x)
            action = rng.binomial(1, pi).astype(np.int8)
            observed = action == 1
            y_obs = np.where(observed, y1, np.nan)

            clients.append(
                ClientBatch(
                    client_id=cid,
                    x=x,
                    action=action,
                    y_obs=y_obs,
                    observed=observed,
                    propensity=pi,
                    y1=y1,
                    y0=y0,
                )
            )
            xs.append(x)
            y1s.append(y1)
            y0s.append(y0)

        return FederatedDataset(
            clients=clients,
            x_all=np.vstack(xs),
            y1_all=np.concatenate(y1s),
            y0_all=np.concatenate(y0s),
            target_action=1,
        )


class ACSIncomeMultistateDGP:
    """
    Multi-client ACS Income-style DGP with site-specific approval policies.

    Uses folktables ACSIncome when available; otherwise a semi-synthetic multi-state
    fallback with the same interface and known Y(1) oracle.
    """

    def __init__(
        self,
        n_clients: int = 5,
        n_per_client: int = 1500,
        seed: int = 42,
        policy_overlap: str = "partial",
        policy_strength: float = 2.5,
    ) -> None:
        if n_clients not in (5, 10):
            raise ValueError("acs_income_multistate supports K_clients in {5, 10}")
        self.n_clients = n_clients
        self.n_per_client = n_per_client
        self.seed = seed
        self.policy_overlap = policy_overlap
        self.policy_strength = policy_strength
        self.data_source = "pending"

    def _load_acs_pool(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (features, state_ids, income_labels)."""
        if _FOLKTABLES_AVAILABLE and ACSIncome is not None:
            try:
                states = ACSIncome.states[: self.n_clients]
                features_list: List[np.ndarray] = []
                labels_list: List[np.ndarray] = []
                state_ids_list: List[np.ndarray] = []
                for sid, st in enumerate(states):
                    feat, lab, _ = ACSIncome.load_data(
                        state=st,
                        root_dir=None,
                        download=True,
                    )
                    if feat.shape[0] < 500:
                        continue
                    n_take = min(3000, feat.shape[0])
                    idx = rng.choice(feat.shape[0], size=n_take, replace=False)
                    features_list.append(feat[idx].astype(float))
                    labels_list.append(lab[idx].astype(float))
                    state_ids_list.append(np.full(n_take, sid, dtype=int))
                if features_list:
                    x = np.vstack(features_list)
                    y_inc = np.concatenate(labels_list)
                    state_ids = np.concatenate(state_ids_list)
                    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
                    self.data_source = "folktables_acs"
                    return x, state_ids, y_inc
            except Exception:
                pass

        # Semi-synthetic multi-state fallback (K site clusters).
        n_pool = max(8000, self.n_clients * self.n_per_client)
        x = rng.normal(size=(n_pool, 8))
        state_ids = rng.integers(0, self.n_clients, size=n_pool)
        for sid in range(self.n_clients):
            mask = state_ids == sid
            x[mask, 0] += 0.8 * sid - 0.4 * self.n_clients
            x[mask, 1] += 0.5 * np.sin(sid)
        y_inc = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(float)
        self.data_source = "semi_synthetic_fallback"
        return x, state_ids, y_inc

    def _outcome_models(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        score = (
            0.6 * x[:, 0]
            - 0.35 * x[:, 1]
            + 0.15 * x[:, 2]
            + 0.1 * x[:, 3] ** 2
        )
        y1 = 1.2 + score + 0.25 * (score ** 2) + rng.normal(0.0, 0.45, size=x.shape[0])
        y0 = 0.25 * score + rng.normal(0.0, 0.45, size=x.shape[0])
        return y1, y0

    def _client_policy(self, client_id: int, x: np.ndarray) -> np.ndarray:
        angle = 2.0 * np.pi * client_id / self.n_clients
        partial = self.policy_strength * (
            x[:, 0] * np.cos(angle) + x[:, 1] * np.sin(angle)
        )
        logits = _overlap_policy_logits(
            client_id,
            self.n_clients,
            x,
            self.policy_overlap,
            partial,
            strength=self.policy_strength,
        )
        return _sigmoid(logits)

    def sample(self) -> FederatedDataset:
        rng = np.random.default_rng(self.seed)
        x_pool, state_ids, _ = self._load_acs_pool(rng)
        y1_pool, y0_pool = self._outcome_models(x_pool, rng)

        clients: List[ClientBatch] = []
        xs: List[np.ndarray] = []
        y1s: List[np.ndarray] = []
        y0s: List[np.ndarray] = []

        for cid in range(self.n_clients):
            if self.data_source == "folktables_acs":
                mask = state_ids == cid
                if mask.sum() < self.n_per_client:
                    idx = rng.choice(x_pool.shape[0], size=self.n_per_client, replace=True)
                else:
                    pool_idx = np.where(mask)[0]
                    idx = rng.choice(pool_idx, size=self.n_per_client, replace=False)
            else:
                site_mask = state_ids == cid
                if site_mask.sum() >= self.n_per_client:
                    pool_idx = np.where(site_mask)[0]
                    idx = rng.choice(pool_idx, size=self.n_per_client, replace=False)
                else:
                    idx = rng.choice(x_pool.shape[0], size=self.n_per_client, replace=True)

            x = x_pool[idx]
            y1 = y1_pool[idx]
            y0 = y0_pool[idx]
            pi = self._client_policy(cid, x)
            action = rng.binomial(1, pi).astype(np.int8)
            observed = action == 1
            y_obs = np.where(observed, y1, np.nan)

            clients.append(
                ClientBatch(
                    client_id=cid,
                    x=x,
                    action=action,
                    y_obs=y_obs,
                    observed=observed,
                    propensity=pi,
                    y1=y1,
                    y0=y0,
                )
            )
            xs.append(x)
            y1s.append(y1)
            y0s.append(y0)

        return FederatedDataset(
            clients=clients,
            x_all=np.vstack(xs),
            y1_all=np.concatenate(y1s),
            y0_all=np.concatenate(y0s),
            target_action=1,
        )


LAST_T2_SOURCE = "unknown"


def make_dgp(
    dgp_id: str,
    seed: int,
    n_per_client: int = 2000,
    policy_overlap: str = "partial",
    n_clients: Optional[int] = None,
) -> FederatedDataset:
    """Factory for PROTOCOL DGP ids with stress-axis parameters."""
    global LAST_T2_SOURCE
    if dgp_id == "two_client_complementary":
        k = n_clients or 2
        return TwoClientComplementaryDGP(
            n_per_client=n_per_client,
            seed=seed,
            policy_overlap=policy_overlap,
            n_clients=k,
        ).sample()
    if dgp_id == "semi_synthetic_credit":
        k = n_clients or 2
        return SemiSyntheticCreditDGP(
            n_per_client=n_per_client,
            seed=seed + 1,
            policy_overlap=policy_overlap,
            n_clients=k,
        ).sample()
    if dgp_id == "acs_income_multistate":
        k = n_clients or 5
        dgp = ACSIncomeMultistateDGP(
            n_clients=k,
            n_per_client=n_per_client,
            seed=seed + 2,
            policy_overlap=policy_overlap,
        )
        data = dgp.sample()
        LAST_T2_SOURCE = dgp.data_source
        return data
    raise ValueError(f"Unknown DGP id: {dgp_id}")


def client_observed_means(data: FederatedDataset) -> Dict[str, float]:
    """Diagnostic: naive mean of observed outcomes per client."""
    out: Dict[str, float] = {}
    for c in data.clients:
        mask = c.observed
        out[f"client_{c.client_id}_naive_mean"] = float(np.nanmean(c.y_obs[mask]))
    return out
