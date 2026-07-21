"""Decision-censored federated outcomes — Idea 1."""

from .dgp import TwoClientComplementaryDGP, SemiSyntheticCreditDGP
from .baselines_fl import estimate_fedprox, estimate_fedpu_adapted, estimate_scaffold
from .estimators import (
    estimate_chen_icml2025_pooled_dr,
    estimate_fed_aipw_xiong,
    estimate_local_ipw,
    estimate_federated_ipw,
    estimate_federated_dr,
    estimate_naive_fedavg,
    estimate_local_only,
    run_protocol_estimators,
)
from .metrics import counterfactual_risk_oracle, bias_vs_oracle, evaluate_kill_gate
from .protocol import FederatedMomentProtocol

__all__ = [
    "TwoClientComplementaryDGP",
    "SemiSyntheticCreditDGP",
    "estimate_local_ipw",
    "estimate_federated_ipw",
    "estimate_federated_dr",
    "estimate_chen_icml2025_pooled_dr",
    "estimate_fed_aipw_xiong",
    "estimate_naive_fedavg",
    "estimate_local_only",
    "estimate_fedprox",
    "estimate_scaffold",
    "estimate_fedpu_adapted",
    "run_protocol_estimators",
    "counterfactual_risk_oracle",
    "bias_vs_oracle",
    "evaluate_kill_gate",
    "FederatedMomentProtocol",
]
