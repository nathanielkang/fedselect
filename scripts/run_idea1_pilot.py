#!/usr/bin/env python3
"""E2 pilot runner — all PROTOCOL methods on T0+T1 DGPs (local only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.idea1_censored.dgp import SemiSyntheticCreditDGP, TwoClientComplementaryDGP
from src.idea1_censored.estimators import run_protocol_estimators
from src.idea1_censored.metrics import bias_vs_oracle, write_json

PILOT_METHODS = [
    "local_only",
    "naive_fedavg",
    "fedprox",
    "scaffold",
    "local_ipw",
    "federated_ipw",
    "fedpu_adapted",
    "federated_dr",
]

DGP_FACTORIES = {
    "two_client_complementary": lambda n, seed: TwoClientComplementaryDGP(
        n_per_client=n, seed=seed
    ).sample(),
    "semi_synthetic_credit": lambda n, seed: SemiSyntheticCreditDGP(
        n_per_client=n, seed=seed + 1
    ).sample(),
}


def _rank_methods(rows: pd.DataFrame) -> Dict[str, Any]:
    """Mean |bias| rank across DGP x seed cells (lower is better)."""
    agg = (
        rows.groupby("method")["abs_bias"]
        .mean()
        .sort_values()
    )
    ranks = {}
    for rank, (method, mean_bias) in enumerate(agg.items(), start=1):
        ranks[method] = {
            "mean_abs_bias": float(mean_bias),
            "mean_rank": float(rank),
        }
    return ranks


def _evaluate_pilot_gate(rows: pd.DataFrame) -> Dict[str, Any]:
    reasons: List[str] = []
    passed = True

    finite_ok = rows["estimate"].apply(np.isfinite).all()
    if finite_ok:
        reasons.append("All methods emit finite estimates")
    else:
        passed = False
        bad = rows[~rows["estimate"].apply(np.isfinite)][["dgp", "method"]]
        reasons.append(f"FAIL: non-finite estimates in {len(bad)} cells")

    comp = rows[rows["dgp"] == "two_client_complementary"]
    if comp.empty:
        passed = False
        reasons.append("FAIL: missing complementary DGP rows")
    else:
        mean_bias = comp.groupby("method")["abs_bias"].mean().sort_values()
        top2 = list(mean_bias.index[:2])
        if "federated_dr" in top2:
            reasons.append(
                f"federated_dr in top-2 on complementary DGP "
                f"(rank {list(mean_bias.index).index('federated_dr') + 1}, "
                f"top2={top2})"
            )
        else:
            passed = False
            dr_rank = list(mean_bias.index).index("federated_dr") + 1
            reasons.append(
                f"FAIL: federated_dr rank {dr_rank} on complementary DGP "
                f"(not top-2); order={list(mean_bias.index)}"
            )

    return {"pass": passed, "reasons": reasons}


def write_pilot_report(
    path: Path,
    rows: pd.DataFrame,
    rank_summary: Dict[str, Any],
    gate: Dict[str, Any],
    seeds: List[int],
) -> None:
    lines = [
        "# Idea 1 E2 Pilot Report",
        "",
        f"**Overall PASS:** `{gate['pass']}`",
        f"**Seeds:** {seeds}",
        "",
        "## Pilot gate",
        "",
    ]
    for r in gate.get("reasons", []):
        lines.append(f"- {r}")

    lines.extend(["", "## Mean |bias| ranks (all DGPs x seeds)", ""])
    sorted_methods = sorted(
        rank_summary.items(),
        key=lambda kv: kv[1]["mean_rank"],
    )
    for method, info in sorted_methods:
        lines.append(
            f"- {method}: rank={info['mean_rank']:.0f}, "
            f"mean |bias|={info['mean_abs_bias']:.4f}"
        )

    lines.extend(["", "## Per-cell results", ""])
    for (dgp, seed), grp in rows.groupby(["dgp", "seed"]):
        oracle = grp["oracle"].iloc[0]
        lines.append(f"### {dgp} (seed={seed}), oracle={oracle:.4f}")
        sub = grp.sort_values("abs_bias")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['method']}: est={row['estimate']:.4f}, |bias|={row['abs_bias']:.4f}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Idea 1 E2 pilot (T0+T1)")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--n", type=int, default=2000, help="samples per client")
    parser.add_argument("--n-rounds", type=int, default=2, dest="n_rounds")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/idea1_pilot",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for dgp_name, factory in DGP_FACTORIES.items():
            data = factory(args.n, seed)
            oracle = data.oracle_risk()
            estimates = run_protocol_estimators(
                data, seed=seed, dp_eps=None, n_rounds=args.n_rounds
            )
            for method in PILOT_METHODS:
                est = estimates[method]
                rows.append(
                    {
                        "seed": seed,
                        "dgp": dgp_name,
                        "method": method,
                        "estimate": est,
                        "oracle": oracle,
                        "bias": est - oracle,
                        "abs_bias": bias_vs_oracle(est, oracle),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pilot_results.csv", index=False)

    rank_summary = _rank_methods(df)
    write_json(out_dir / "rank_summary.json", rank_summary)

    gate = _evaluate_pilot_gate(df)
    gate["phase"] = "E2"
    gate["seeds"] = seeds
    gate["dgps"] = list(DGP_FACTORIES.keys())
    write_json(out_dir / "pilot_gate.json", gate)

    write_pilot_report(
        out_dir / "IDEA1_PILOT_REPORT.md",
        df,
        rank_summary,
        gate,
        seeds,
    )

    print(f"Pilot PASS={gate['pass']}")
    print("Mean |bias| ranks:")
    for method, info in sorted(rank_summary.items(), key=lambda kv: kv[1]["mean_rank"]):
        print(
            f"  {info['mean_rank']:.0f}. {method}: "
            f"mean |bias|={info['mean_abs_bias']:.4f}"
        )
    print(f"Wrote {out_dir}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
