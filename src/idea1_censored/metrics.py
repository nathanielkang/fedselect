"""Metrics, kill gates, and reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dgp import FederatedDataset


def counterfactual_risk_oracle(data: FederatedDataset, action: int = 1) -> float:
    return data.oracle_risk(action=action)


def bias_vs_oracle(estimate: float, oracle: float) -> float:
    if np.isnan(estimate):
        return float("inf")
    return float(abs(estimate - oracle))


def interval_width(interval: Optional[Tuple[float, float]]) -> Optional[float]:
    if interval is None:
        return None
    return float(interval[1] - interval[0])


def evaluate_kill_gate(
    biases: Dict[str, float],
    ratio_threshold: float = 3.0,
    abs_dr_cap: float = 0.35,
) -> Dict[str, Any]:
    """
    PASS if federated DR bias is much smaller than FedAvg and all local methods.

    ratio_threshold: fed_dr must be <= bias / ratio_threshold vs comparators.
    abs_dr_cap: |bias_fed_dr| must also be below this absolute cap.
    """
    b_dr = biases.get("federated_dr", float("inf"))
    b_avg = biases.get("naive_fedavg", float("inf"))
    local_keys = [k for k in biases if k.startswith("local_")]
    b_local_max = max((biases[k] for k in local_keys), default=float("inf"))

    reasons: List[str] = []
    pass_dr_vs_avg = b_dr <= b_avg / ratio_threshold
    pass_dr_vs_local = b_dr <= b_local_max / ratio_threshold
    pass_abs = b_dr <= abs_dr_cap

    if pass_dr_vs_avg:
        reasons.append(f"federated_dr bias ({b_dr:.4f}) << naive_fedavg ({b_avg:.4f})")
    else:
        reasons.append(
            f"FAIL: federated_dr bias ({b_dr:.4f}) not << naive_fedavg ({b_avg:.4f})"
        )
    if pass_dr_vs_local:
        reasons.append(
            f"federated_dr bias ({b_dr:.4f}) << worst local ({b_local_max:.4f})"
        )
    else:
        reasons.append(
            f"FAIL: federated_dr bias ({b_dr:.4f}) not << worst local ({b_local_max:.4f})"
        )
    if pass_abs:
        reasons.append(f"|bias_fed_dr| <= {abs_dr_cap}")
    else:
        reasons.append(f"FAIL: |bias_fed_dr|={b_dr:.4f} exceeds cap {abs_dr_cap}")

    passed = pass_dr_vs_avg and pass_dr_vs_local and pass_abs

    return {
        "idea": 1,
        "pass": passed,
        "reasons": reasons,
        "metrics": {
            "bias_federated_dr": b_dr,
            "bias_naive_fedavg": b_avg,
            "bias_worst_local": b_local_max,
            "bias_local_only_max": max(
                (biases.get(k, 0.0) for k in biases if k.startswith("local_only")),
                default=float("nan"),
            ),
            "bias_local_ipw_max": max(
                (biases.get(k, 0.0) for k in biases if k.startswith("local_ipw")),
                default=float("nan"),
            ),
            "bias_federated_ipw": biases.get("federated_ipw", float("nan")),
            "ratio_threshold": ratio_threshold,
            "abs_dr_cap": abs_dr_cap,
        },
    }


def summarize_experiment(
    dgp_name: str,
    estimates: Dict[str, float],
    oracle: float,
    dp_eps: Optional[float] = None,
    interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    biases = {k: bias_vs_oracle(v, oracle) for k, v in estimates.items() if k != "oracle"}
    gate = evaluate_kill_gate(biases)
    gate["metrics"]["oracle"] = oracle
    gate["metrics"]["dgp"] = dgp_name
    gate["metrics"]["estimates"] = estimates
    gate["metrics"]["dp_eps"] = dp_eps
    if interval is not None:
        gate["metrics"]["dp_interval"] = list(interval)
        gate["metrics"]["dp_interval_width"] = interval_width(interval)
    return gate


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_smoke_report(
    path: Path,
    results: List[Dict[str, Any]],
    overall_gate: Dict[str, Any],
) -> None:
    lines = [
        "# Idea 1 Smoke Report — Decision-Censored Federated Outcomes",
        "",
        f"**Overall PASS:** `{overall_gate['pass']}`",
        "",
        "## Kill gate",
        "",
    ]
    for r in overall_gate.get("reasons", []):
        lines.append(f"- {r}")
    lines.extend(["", "## Per-DGP results", ""])
    for res in results:
        m = res["metrics"]
        lines.append(f"### {m.get('dgp', 'unknown')}")
        lines.append(f"- Oracle E[Y(1)]: {m.get('oracle', float('nan')):.4f}")
        est = m.get("estimates", {})
        for k, v in sorted(est.items()):
            if k == "oracle":
                continue
            b = abs(v - m.get("oracle", 0.0))
            lines.append(f"- {k}: {v:.4f} (|bias|={b:.4f})")
        if m.get("dp_interval_width") is not None:
            lines.append(f"- DP interval width: {m['dp_interval_width']:.4f}")
        lines.append(f"- Gate pass: `{res['pass']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
