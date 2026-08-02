#!/usr/bin/env python3
"""V2+DP rows for Fig. 5 — federated_dr at eps=2, partial overlap, 3 DGPs x 5 seeds."""

from __future__ import annotations

import argparse
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
from src.idea1_censored.estimators import estimate_federated_dr, run_protocol_estimators
from src.idea1_censored.metrics import bias_vs_oracle, interval_width

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DP_EPS = 2.0

# Fig. 5 protocol: partial overlap, one cell per tier (matches V2 non-DP partial rows)
FIG5_DP_SPECS: List[Dict[str, Any]] = [
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "partial"},
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "partial"},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "partial"},
]


def run_dp_matrix(
    seeds: List[int],
    n_per_client: int,
    n_rounds: int,
) -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []
    t2_source = "unknown"

    for seed in seeds:
        for spec in FIG5_DP_SPECS:
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
            estimates = run_protocol_estimators(
                data, seed=seed, dp_eps=DP_EPS, n_rounds=n_rounds
            )
            dr_res = estimate_federated_dr(data, seed=seed, dp_eps=DP_EPS)
            dr_width = interval_width(dr_res.interval)

            est = estimates["federated_dr"]
            rows.append(
                {
                    "seed": seed,
                    "dgp": spec["dgp"],
                    "tier": spec["tier"],
                    "policy_overlap": spec["policy_overlap"],
                    "K_clients": spec["K_clients"],
                    "dp_eps": DP_EPS,
                    "method": "federated_dr",
                    "estimate": est,
                    "oracle": oracle,
                    "theta_star": oracle,
                    "bias": est - oracle,
                    "abs_bias": bias_vs_oracle(est, oracle),
                    "dp_interval_width": dr_width,
                }
            )

    return pd.DataFrame(rows), t2_source


def merge_into_v2_csv(
    base_csv: Path,
    dp_df: pd.DataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    base = pd.read_csv(base_csv)
    if "dp_interval_width" not in base.columns:
        base["dp_interval_width"] = np.nan

    base["dp_eps"] = pd.to_numeric(base["dp_eps"], errors="coerce")
    dp_df = dp_df.copy()
    dp_df["dp_eps"] = float(DP_EPS)

    # Drop any prior V2 federated_dr dp_eps=2 rows for the same keys (idempotent re-run)
    keys = ["seed", "dgp", "tier", "policy_overlap", "K_clients", "dp_eps", "method"]
    mask_dp_old = (
        (base["method"] == "federated_dr")
        & (base["dp_eps"] == DP_EPS)
        & base["tier"].isin(["T0", "T1", "T2"])
    )
    base = base[~mask_dp_old].copy()

    merged = pd.concat([base, dp_df], ignore_index=True)
    merged.to_csv(out_csv, index=False)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="V2+DP Fig.5 rows merged into v2_baselines CSV")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=2000, help="samples per client")
    parser.add_argument("--n-rounds", type=int, default=2, dest="n_rounds")
    parser.add_argument(
        "--base-csv",
        type=str,
        default="results/v2_baselines/results.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/v2_baselines/results.csv",
        help="Merged output (default: overwrite base with DP rows appended)",
    )
    parser.add_argument(
        "--dp-only-csv",
        type=str,
        default="results/v2_baselines/dp_fig5_rows.csv",
        help="Also write standalone DP rows",
    )
    parser.add_argument("--smoke", action="store_true", help="1 seed, smaller n")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    n_per_client = min(args.n, 800) if args.smoke else args.n
    if args.smoke:
        seeds = seeds[:1] or [42]

    base_csv = ROOT / args.base_csv
    out_csv = ROOT / args.out_csv
    dp_only = ROOT / args.dp_only_csv
    dp_only.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    dp_df, t2_source = run_dp_matrix(seeds, n_per_client, args.n_rounds)
    wall_sec = time.perf_counter() - t0

    dp_df.to_csv(dp_only, index=False)
    merged = merge_into_v2_csv(base_csv, dp_df, out_csv)

    finite_ok = dp_df["estimate"].apply(np.isfinite).all()
    n_dp = len(dp_df)
    print(f"V2+DP smoke={args.smoke} finite={finite_ok} wall={wall_sec:.1f}s")
    print(f"T2 source={t2_source}")
    print(f"Wrote {dp_only} ({n_dp} rows)")
    print(f"Merged into {out_csv} (total {len(merged)} rows)")
    return 0 if finite_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
