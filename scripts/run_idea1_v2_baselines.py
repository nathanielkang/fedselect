#!/usr/bin/env python3
"""V2 recent baselines matrix — Chen pooled DR + Fed-AIPW (Xiong-style)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.idea1_censored.dgp import make_dgp
from src.idea1_censored.estimators import run_protocol_estimators
from src.idea1_censored.metrics import bias_vs_oracle

V2_METHODS = [
    "naive_fedavg",
    "federated_ipw",
    "federated_dr",
    "chen_icml2025_pooled_dr",
    "fed_aipw_xiong",
    "local_only",
    "local_ipw",
    "fedprox",
    "scaffold",
    "fedpu_adapted",
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]

V2_CELL_SPECS: List[Dict[str, Any]] = [
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "partial"},
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "full"},
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "partial"},
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "full"},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "partial"},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "full"},
]

SMOKE_CELL_SPECS = [
    c for c in V2_CELL_SPECS if c["tier"] == "T0" and c["policy_overlap"] == "partial"
]


def _aggregate_by_method_tier(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean |bias| grouped by method and tier."""
    return (
        rows.groupby(["method", "tier"])["abs_bias"]
        .mean()
        .reset_index()
        .rename(columns={"abs_bias": "mean_abs_bias"})
    )


def _pivot_summary(rows: pd.DataFrame) -> pd.DataFrame:
    agg = _aggregate_by_method_tier(rows)
    pivot = agg.pivot(index="method", columns="tier", values="mean_abs_bias")
    for tier in ("T0", "T1", "T2"):
        if tier not in pivot.columns:
            pivot[tier] = np.nan
    pivot["aggregate"] = rows.groupby("method")["abs_bias"].mean()
    return pivot.sort_values("aggregate")


def write_summary_txt(path: Path, pivot: pd.DataFrame) -> None:
    lines = ["# V2 baselines — mean |bias| by method x tier", ""]
    cols = ["T0", "T1", "T2", "aggregate"]
    header = "| Method | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
    lines.extend([header, sep])
    for method, row in pivot.iterrows():
        cells = [f"{row.get(c, float('nan')):.4f}" if np.isfinite(row.get(c, np.nan)) else "—" for c in cols]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(path: Path, pivot: pd.DataFrame) -> None:
    payload = {
        "mean_abs_bias": {
            method: {col: float(row[col]) if np.isfinite(row[col]) else None for col in pivot.columns}
            for method, row in pivot.iterrows()
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_matrix(
    seeds: List[int],
    cell_specs: List[Dict[str, Any]],
    n_per_client: int,
    n_rounds: int,
) -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []
    t2_source = "unknown"

    for seed in seeds:
        for spec in cell_specs:
            data = make_dgp(
                spec["dgp"],
                seed=seed,
                n_per_client=n_per_client,
                policy_overlap=spec["policy_overlap"],
                n_clients=spec["K_clients"],
            )
            if spec["dgp"] == "acs_income_multistate":
                from src.idea1_censored.dgp import LAST_T2_SOURCE

                t2_source = LAST_T2_SOURCE

            oracle = data.oracle_risk()
            estimates = run_protocol_estimators(data, seed=seed, dp_eps=None, n_rounds=n_rounds)

            for method in V2_METHODS:
                est = estimates[method]
                rows.append(
                    {
                        "seed": seed,
                        "dgp": spec["dgp"],
                        "tier": spec["tier"],
                        "policy_overlap": spec["policy_overlap"],
                        "K_clients": spec["K_clients"],
                        "dp_eps": None,
                        "method": method,
                        "estimate": est,
                        "oracle": oracle,
                        "theta_star": oracle,
                        "bias": est - oracle,
                        "abs_bias": bias_vs_oracle(est, oracle),
                    }
                )

    return pd.DataFrame(rows), t2_source


def main() -> int:
    parser = argparse.ArgumentParser(description="FedSelect V2 recent baselines matrix")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=2000, help="samples per client")
    parser.add_argument("--n-rounds", type=int, default=2, dest="n_rounds")
    parser.add_argument("--out-dir", type=str, default="results/v2_baselines")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="T0 partial only, 1 seed, smaller n",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cell_specs = SMOKE_CELL_SPECS if args.smoke else V2_CELL_SPECS
    n_per_client = min(args.n, 800) if args.smoke else args.n
    if args.smoke:
        seeds = seeds[:1] or [42]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df, t2_source = run_matrix(seeds, cell_specs, n_per_client, args.n_rounds)
    wall_sec = time.perf_counter() - t0

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    pivot = _pivot_summary(df)
    write_summary_txt(out_dir / "summary_means.txt", pivot)
    write_summary_json(out_dir / "summary_means.json", pivot)

    finite_ok = df["estimate"].apply(np.isfinite).all()
    print(f"V2 baselines smoke={args.smoke} finite={finite_ok} wall={wall_sec:.1f}s")
    print(f"T2 source={t2_source}")
    print(f"Wrote {csv_path}")
    print("\nMean |bias| (methods x tier):")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    return 0 if finite_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
