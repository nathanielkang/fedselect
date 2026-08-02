#!/usr/bin/env python3
"""LOCO on/off ablation for FedSelect federated_dr (P1.8)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.idea1_censored.dgp import make_dgp
from src.idea1_censored.estimators import (
    estimate_federated_dr,
    estimate_federated_dr_no_loco,
)
from src.idea1_censored.metrics import bias_vs_oracle

DEFAULT_SEEDS = [42, 43, 44, 45, 46]

CELL_SPECS: List[Dict[str, Any]] = [
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "partial"},
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "full"},
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "partial"},
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "full"},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "partial"},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "full"},
]

SMOKE_CELL_SPECS = [
    c for c in CELL_SPECS if c["tier"] == "T0" and c["policy_overlap"] == "partial"
]

METHODS = {
    "federated_dr": estimate_federated_dr,
    "federated_dr_no_loco": estimate_federated_dr_no_loco,
}


def _pivot_summary(rows: pd.DataFrame) -> pd.DataFrame:
    agg = (
        rows.groupby(["method", "tier"])["abs_bias"]
        .mean()
        .reset_index()
        .rename(columns={"abs_bias": "mean_abs_bias"})
    )
    pivot = agg.pivot(index="method", columns="tier", values="mean_abs_bias")
    for tier in ("T0", "T1", "T2"):
        if tier not in pivot.columns:
            pivot[tier] = np.nan
    pivot["aggregate"] = rows.groupby("method")["abs_bias"].mean()
    return pivot.sort_values("aggregate")


def write_summary_md(path: Path, pivot: pd.DataFrame, note: str) -> None:
    lines = [
        "# LOCO ablation — mean |bias| by method x tier",
        "",
        note,
        "",
        "| Method | T0 | T1 | T2 | aggregate |",
        "|---|---|---|---|---|",
    ]
    for method, row in pivot.iterrows():
        cells = [
            f"{row.get(c, float('nan')):.4f}" if np.isfinite(row.get(c, np.nan)) else "—"
            for c in ("T0", "T1", "T2", "aggregate")
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_matrix(
    seeds: List[int],
    cell_specs: List[Dict[str, Any]],
    n_per_client: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for spec in cell_specs:
            data = make_dgp(
                spec["dgp"],
                seed=seed,
                n_per_client=n_per_client,
                policy_overlap=spec["policy_overlap"],
                n_clients=spec["K_clients"],
            )
            oracle = data.oracle_risk()
            for method_name, est_fn in METHODS.items():
                est = est_fn(data, seed=seed).estimate
                rows.append(
                    {
                        "seed": seed,
                        "dgp": spec["dgp"],
                        "tier": spec["tier"],
                        "policy_overlap": spec["policy_overlap"],
                        "K_clients": spec["K_clients"],
                        "method": method_name,
                        "estimate": est,
                        "oracle": oracle,
                        "bias": est - oracle,
                        "abs_bias": bias_vs_oracle(est, oracle),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="FedSelect LOCO on/off ablation")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=2000, help="samples per client")
    parser.add_argument("--out-dir", type=str, default="results/loco_ablation")
    parser.add_argument("--smoke", action="store_true", help="T0 partial, 1 seed, n=800")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cell_specs = SMOKE_CELL_SPECS if args.smoke else CELL_SPECS
    n_per_client = min(args.n, 800) if args.smoke else args.n
    if args.smoke:
        seeds = seeds[:1] or [42]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    note = (
        "Ablation: `federated_dr` = LOCO on (leave-one-client-out shared mu); "
        "`federated_dr_no_loco` = global FedAvg bridge including own client "
        "(same clip eta=0.05, rich `_expand_design`, truncated IPW). "
        "Non-DP partial+full overlap cells; seeds 42–46."
    )

    t0 = time.perf_counter()
    df = run_matrix(seeds, cell_specs, n_per_client)
    wall_sec = time.perf_counter() - t0

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    pivot = _pivot_summary(df)
    write_summary_md(out_dir / "summary.md", pivot, note)
    pivot.to_json(out_dir / "summary.json", indent=2, orient="index")

    finite_ok = df["estimate"].apply(np.isfinite).all()
    print(f"LOCO ablation smoke={args.smoke} finite={finite_ok} wall={wall_sec:.1f}s")
    print(f"Wrote {csv_path}")
    print("\nMean |bias|:")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    return 0 if finite_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
