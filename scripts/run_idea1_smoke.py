#!/usr/bin/env python3
"""Local smoke test for Idea 1 — decision-censored federated outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.idea1_censored.dgp import SemiSyntheticCreditDGP, TwoClientComplementaryDGP
from src.idea1_censored.estimators import (
    estimate_chen_icml2025_pooled_dr,
    estimate_fed_aipw_xiong,
    estimate_federated_dr,
    estimate_federated_ipw,
    estimate_local_ipw,
    estimate_local_only,
    estimate_naive_fedavg,
)
from src.idea1_censored.metrics import (
    bias_vs_oracle,
    summarize_experiment,
    write_json,
    write_smoke_report,
)
from src.idea1_censored.protocol import FederatedMomentProtocol


def run_dgp(
    name: str,
    data,
    seed: int,
    dp_eps: float | None,
) -> dict:
    oracle = data.oracle_risk()
    estimates = {"oracle": oracle}
    estimates["naive_fedavg"] = estimate_naive_fedavg(data).estimate
    for r in estimate_local_only(data):
        estimates[r.method] = r.estimate
    for r in estimate_local_ipw(data, seed=seed):
        estimates[r.method] = r.estimate
    fed_ipw = estimate_federated_ipw(data, seed=seed, dp_eps=dp_eps)
    fed_dr = estimate_federated_dr(data, seed=seed, dp_eps=dp_eps)
    estimates["federated_ipw"] = fed_ipw.estimate
    estimates["federated_dr"] = fed_dr.estimate
    estimates["chen_icml2025_pooled_dr"] = estimate_chen_icml2025_pooled_dr(
        data, seed=seed
    ).estimate
    estimates["fed_aipw_xiong"] = estimate_fed_aipw_xiong(data, seed=seed).estimate

    proto = FederatedMomentProtocol(n_rounds=2, dp_eps=dp_eps, seed=seed)
    estimates["federated_dr_protocol"] = proto.run_dr_rounds(data).estimate

    return summarize_experiment(
        dgp_name=name,
        estimates=estimates,
        oracle=oracle,
        dp_eps=dp_eps,
        interval=fed_dr.interval,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Idea 1 smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=2000, help="samples per client")
    parser.add_argument("--dp-eps", type=float, default=None, dest="dp_eps")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/idea1_smoke",
        help="output directory relative to 2_Code root",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dgps = [
        (
            "two_client_complementary",
            TwoClientComplementaryDGP(n_per_client=args.n, seed=args.seed).sample(),
        ),
        (
            "semi_synthetic_credit",
            SemiSyntheticCreditDGP(n_per_client=args.n, seed=args.seed + 1).sample(),
        ),
    ]

    results = []
    rows = []
    for name, data in dgps:
        res = run_dgp(name, data, seed=args.seed, dp_eps=args.dp_eps)
        results.append(res)
        oracle = res["metrics"]["oracle"]
        for method, est in res["metrics"]["estimates"].items():
            if method == "oracle":
                continue
            rows.append(
                {
                    "dgp": name,
                    "method": method,
                    "estimate": est,
                    "oracle": oracle,
                    "bias": bias_vs_oracle(est, oracle),
                }
            )

    overall_pass = any(r["pass"] for r in results)
    overall = {
        "idea": 1,
        "pass": overall_pass,
        "reasons": [
            "PASS if any DGP regime satisfies kill gate"
            if overall_pass
            else "FAIL: no DGP passed kill gate",
        ],
        "regimes": results,
    }

    write_json(out_dir / "idea1_kill_gate.json", overall)
    pd.DataFrame(rows).to_csv(out_dir / "idea1_smoke_results.csv", index=False)
    write_smoke_report(out_dir / "IDEA1_SMOKE_REPORT.md", results, overall)

    print(f"Kill gate PASS={overall_pass}")
    for r in results:
        m = r["metrics"]
        print(
            f"  [{m['dgp']}] oracle={m['oracle']:.4f} "
            f"fed_dr_bias={m['bias_federated_dr']:.4f} "
            f"fedavg_bias={m['bias_naive_fedavg']:.4f} pass={r['pass']}"
        )
    print(f"Wrote {out_dir / 'IDEA1_SMOKE_REPORT.md'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
