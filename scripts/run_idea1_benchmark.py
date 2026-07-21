#!/usr/bin/env python3
"""Slightly larger grid for GCP benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_idea1_smoke import run_dgp
from src.idea1_censored.dgp import SemiSyntheticCreditDGP, TwoClientComplementaryDGP
from src.idea1_censored.metrics import write_json, write_smoke_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Idea 1 benchmark grid")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--dp-eps", type=float, default=1.0, dest="dp_eps")
    parser.add_argument("--out-dir", type=str, default="results/idea1_benchmark")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    all_results = []
    rows = []
    for seed in seeds:
        for name, factory in [
            ("two_client_complementary", lambda s: TwoClientComplementaryDGP(n_per_client=args.n, seed=s).sample()),
            ("semi_synthetic_credit", lambda s: SemiSyntheticCreditDGP(n_per_client=args.n, seed=s + 1).sample()),
        ]:
            data = factory(seed)
            res = run_dgp(name, data, seed=seed, dp_eps=args.dp_eps)
            res["metrics"]["seed"] = seed
            all_results.append(res)
            oracle = res["metrics"]["oracle"]
            for method, est in res["metrics"]["estimates"].items():
                if method == "oracle":
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "dgp": name,
                        "method": method,
                        "estimate": est,
                        "oracle": oracle,
                        "bias": abs(est - oracle),
                    }
                )

    overall_pass = any(r["pass"] for r in all_results)
    payload = {"idea": 1, "pass": overall_pass, "regimes": all_results}
    write_json(out_dir / "idea1_benchmark_kill_gate.json", payload)
    pd.DataFrame(rows).to_csv(out_dir / "idea1_benchmark_results.csv", index=False)
    write_smoke_report(
        out_dir / "IDEA1_BENCHMARK_REPORT.md",
        all_results,
        {"pass": overall_pass, "reasons": payload.get("reasons", [])},
    )
    print(f"Benchmark PASS={overall_pass}; wrote {out_dir}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
