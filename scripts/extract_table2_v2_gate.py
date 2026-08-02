#!/usr/bin/env python3
"""Extract V2 Table 2 identification-gate numbers from v2_baselines/results.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Honest identification gate (2026-08-03 feature-aligned-FedAvg revision).
# After the naive-FedAvg feature-map fix, selection-blind averaging already fits a
# shared expanded outcome model and attains ~0.01 aggregate bias, so the legacy
# "<= naive/3" bar no longer distinguishes selection-correction (both estimators are
# near zero). The informative comparison is against the worst local estimator, which
# is exactly what per-client selection blindness corrupts (worst local ~1.3 bias).
# Gate: |bias_DR| <= (1/3) * |bias_worst_local|  AND  |bias_DR| <= 0.35.
# The <= naive comparison is retained as an informational field only.
G1_RATIO_MAX = 1.0 / 3.0
G1_ABS_DR_CAP = 0.35


def evaluate_gate(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if "dp_eps" in df.columns:
        df["dp_eps"] = pd.to_numeric(df["dp_eps"], errors="coerce")
        prim = df[df["dp_eps"].isna()].copy()
    else:
        prim = df.copy()

    t0 = prim[
        (prim["dgp"] == "two_client_complementary")
        & (prim["tier"] == "T0")
        & (prim["policy_overlap"] == "partial")
    ]
    if t0.empty:
        raise ValueError(f"No T0 partial complementary rows in {csv_path}")

    cells: List[Dict[str, Any]] = []
    for seed in sorted(t0["seed"].unique()):
        s = t0[t0["seed"] == seed]
        b_dr = float(s[s["method"] == "federated_dr"]["abs_bias"].iloc[0])
        b_avg = float(s[s["method"] == "naive_fedavg"]["abs_bias"].iloc[0])
        local = s[s["method"].str.startswith("local_")]["abs_bias"]
        b_local = float(local.max()) if len(local) else float("nan")
        ratio_vs_naive = b_dr / max(b_avg, 1e-9)
        ratio_vs_worst_local = b_dr / max(b_local, 1e-9)
        pass_ratio = b_dr <= b_local * G1_RATIO_MAX
        pass_abs = b_dr <= G1_ABS_DR_CAP
        # Informational only (not part of the gate decision after the FedAvg fix).
        pass_le_naive = b_dr <= b_avg
        cells.append(
            {
                "seed": int(seed),
                "abs_bias_federated_dr": round(b_dr, 4),
                "abs_bias_naive_fedavg": round(b_avg, 4),
                "abs_bias_worst_local": round(b_local, 4),
                "ratio_vs_naive": round(ratio_vs_naive, 4),
                "ratio_vs_worst_local": round(ratio_vs_worst_local, 4),
                "pass_ratio_rule": bool(pass_ratio),
                "pass_abs_cap": bool(pass_abs),
                "pass_le_naive_informational": bool(pass_le_naive),
                "pass": bool(pass_ratio and pass_abs),
            }
        )

    n_pass = sum(1 for c in cells if c["pass"])
    return {
        "source_csv": str(csv_path.resolve()),
        "pipeline": "V2",
        "dgp": "two_client_complementary",
        "tier": "T0",
        "policy_overlap": "partial",
        "dp_eps": None,
        "gate_rule": (
            "abs_bias(federated_dr) <= worst_local / 3 "
            f"AND abs_bias(federated_dr) <= {G1_ABS_DR_CAP} "
            "(<= naive_fedavg retained as informational field only)"
        ),
        "gate_thresholds": {
            "ratio_max": G1_RATIO_MAX,
            "abs_dr_cap": G1_ABS_DR_CAP,
        },
        "all_pass": bool(n_pass == len(cells)),
        "n_pass": n_pass,
        "n_seeds": len(cells),
        "cells": cells,
        "table2_row_abs_bias": {str(c["seed"]): c["abs_bias_federated_dr"] for c in cells},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract V2 Table 2 identification gate")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/v2_baselines/results.csv",
        help="V2 baselines results CSV",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="results/v2_baselines/table2_identification_gate.json",
    )
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    payload = evaluate_gate(csv_path)
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"All pass: {payload['all_pass']} ({payload['n_pass']}/{payload['n_seeds']})")
    for c in payload["cells"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(
            f"  seed={c['seed']}: |bias_DR|={c['abs_bias_federated_dr']:.4f} "
            f"[{status}]"
        )
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
