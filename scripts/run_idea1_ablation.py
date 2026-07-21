#!/usr/bin/env python3
"""E3 local multi-seed ablation — all PROTOCOL pilot methods on T0+T1 DGPs."""

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

ABLATION_METHODS = [
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

E3_MEAN_RANK_MAX = 3.0
E3_POSITION_MAX = 3


def _aggregate_rank_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    """Mean |bias| rank across all seed x DGP cells (lower is better)."""
    agg = rows.groupby("method")["abs_bias"].mean().sort_values()
    summary: Dict[str, Any] = {}
    for rank, (method, mean_bias) in enumerate(agg.items(), start=1):
        summary[method] = {
            "mean_abs_bias": float(mean_bias),
            "mean_rank": float(rank),
        }
    return summary


def _per_cell_ranks(rows: pd.DataFrame) -> pd.DataFrame:
    """Rank methods within each (seed, dgp) cell by |bias|."""
    ranked_rows: List[Dict[str, Any]] = []
    for (seed, dgp), grp in rows.groupby(["seed", "dgp"]):
        sub = grp.sort_values("abs_bias").reset_index(drop=True)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            ranked_rows.append(
                {
                    "seed": seed,
                    "dgp": dgp,
                    "method": row["method"],
                    "abs_bias": row["abs_bias"],
                    "cell_rank": rank,
                }
            )
    return pd.DataFrame(ranked_rows)


def _evaluate_e3_gate(
    rows: pd.DataFrame,
    rank_summary: Dict[str, Any],
    cell_ranks: pd.DataFrame,
) -> Dict[str, Any]:
    reasons: List[str] = []
    passed = True

    finite_ok = rows["estimate"].apply(np.isfinite).all()
    if finite_ok:
        reasons.append("All methods emit finite estimates (no crashes)")
    else:
        passed = False
        bad = rows[~rows["estimate"].apply(np.isfinite)][["seed", "dgp", "method"]]
        reasons.append(f"FAIL: non-finite estimates in {len(bad)} cells")

    dr_cell = cell_ranks[cell_ranks["method"] == "federated_dr"]
    dr_mean_cell_rank = float(dr_cell["cell_rank"].mean()) if not dr_cell.empty else float("nan")
    if np.isfinite(dr_mean_cell_rank) and dr_mean_cell_rank <= E3_MEAN_RANK_MAX:
        reasons.append(
            f"federated_dr mean cell rank {dr_mean_cell_rank:.2f} <= {E3_MEAN_RANK_MAX}"
        )
    else:
        passed = False
        reasons.append(
            f"FAIL: federated_dr mean cell rank {dr_mean_cell_rank:.2f} "
            f"> {E3_MEAN_RANK_MAX}"
        )

    dr_info = rank_summary.get("federated_dr", {})
    dr_position = int(dr_info.get("mean_rank", 999))
    if dr_position <= E3_POSITION_MAX:
        reasons.append(
            f"federated_dr aggregate position {dr_position} <= {E3_POSITION_MAX} "
            f"(mean |bias|={dr_info.get('mean_abs_bias', float('nan')):.4f})"
        )
    else:
        passed = False
        reasons.append(
            f"FAIL: federated_dr aggregate position {dr_position} "
            f"> {E3_POSITION_MAX}"
        )

    return {
        "pass": passed,
        "reasons": reasons,
        "federated_dr_mean_cell_rank": dr_mean_cell_rank,
        "federated_dr_aggregate_position": dr_position,
        "gate_thresholds": {
            "mean_cell_rank_max": E3_MEAN_RANK_MAX,
            "aggregate_position_max": E3_POSITION_MAX,
        },
    }


def write_ablation_report(
    path: Path,
    rows: pd.DataFrame,
    rank_summary: Dict[str, Any],
    cell_ranks: pd.DataFrame,
    gate: Dict[str, Any],
    seeds: List[int],
) -> None:
    lines = [
        "# Idea 1 E3 Ablation Report",
        "",
        f"**Overall PASS:** `{gate['pass']}`",
        f"**Seeds:** {seeds}",
        f"**DGPs:** {list(DGP_FACTORIES.keys())}",
        f"**Methods:** {len(ABLATION_METHODS)} pilot-active (excl. oracle)",
        "",
        "## E3 gate (loose)",
        "",
    ]
    for r in gate.get("reasons", []):
        lines.append(f"- {r}")

    lines.extend(
        [
            "",
            f"- federated_dr mean cell rank: **{gate.get('federated_dr_mean_cell_rank', float('nan')):.2f}**",
            f"- federated_dr aggregate position: **{gate.get('federated_dr_aggregate_position', '?')}**",
            "",
            "## Mean |bias| ranks (aggregate over seeds x DGPs)",
            "",
            "| Rank | Method | Mean |bias| |",
            "|------|--------|-------------|",
        ]
    )
    sorted_methods = sorted(rank_summary.items(), key=lambda kv: kv[1]["mean_rank"])
    for method, info in sorted_methods:
        lines.append(
            f"| {info['mean_rank']:.0f} | {method} | {info['mean_abs_bias']:.4f} |"
        )

    lines.extend(["", "## Per-cell federated_dr ranks", ""])
    dr_cells = cell_ranks[cell_ranks["method"] == "federated_dr"].sort_values(
        ["seed", "dgp"]
    )
    for _, row in dr_cells.iterrows():
        lines.append(
            f"- seed={row['seed']}, {row['dgp']}: rank={row['cell_rank']}, "
            f"|bias|={row['abs_bias']:.4f}"
        )

    lines.extend(["", "## Per-cell results (all methods)", ""])
    for (seed, dgp), grp in rows.groupby(["seed", "dgp"]):
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
    parser = argparse.ArgumentParser(description="Idea 1 E3 multi-seed ablation (T0+T1)")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--n", type=int, default=2000, help="samples per client")
    parser.add_argument("--n-rounds", type=int, default=2, dest="n_rounds")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/idea1_ablation_e3",
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
            for method in ABLATION_METHODS:
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
    df.to_csv(out_dir / "ablation_results.csv", index=False)

    rank_summary = _aggregate_rank_summary(df)
    cell_ranks = _per_cell_ranks(df)

    rank_payload = {
        "methods": rank_summary,
        "federated_dr_mean_cell_rank": float(
            cell_ranks[cell_ranks["method"] == "federated_dr"]["cell_rank"].mean()
        ),
        "n_cells": int(df.groupby(["seed", "dgp"]).ngroups),
        "seeds": seeds,
        "dgps": list(DGP_FACTORIES.keys()),
    }
    write_json(out_dir / "rank_summary.json", rank_payload)

    gate = _evaluate_e3_gate(df, rank_summary, cell_ranks)
    gate["phase"] = "E3"
    gate["seeds"] = seeds
    gate["dgps"] = list(DGP_FACTORIES.keys())
    gate["methods"] = ABLATION_METHODS
    write_json(out_dir / "ablation_gate.json", gate)

    write_ablation_report(
        out_dir / "E3_ABLATION_REPORT.md",
        df,
        rank_summary,
        cell_ranks,
        gate,
        seeds,
    )

    print(f"E3 ablation PASS={gate['pass']}")
    print(f"federated_dr mean cell rank={gate['federated_dr_mean_cell_rank']:.2f}")
    print("Aggregate mean |bias| ranks:")
    for method, info in sorted(rank_summary.items(), key=lambda kv: kv[1]["mean_rank"]):
        print(
            f"  {info['mean_rank']:.0f}. {method}: "
            f"mean |bias|={info['mean_abs_bias']:.4f}"
        )
    print(f"Wrote {out_dir}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
