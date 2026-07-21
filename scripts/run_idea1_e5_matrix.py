#!/usr/bin/env python3
"""E5 capped stress matrix — T0/T1/T2 DGPs, overlap, K, DP axes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.idea1_censored.dgp import ACSIncomeMultistateDGP, make_dgp
from src.idea1_censored.estimators import estimate_federated_dr, run_protocol_estimators
from src.idea1_censored.metrics import bias_vs_oracle, interval_width, write_json

E5_METHODS = [
    "local_only",
    "naive_fedavg",
    "fedprox",
    "scaffold",
    "local_ipw",
    "federated_ipw",
    "fedpu_adapted",
    "federated_dr",
]

E5_MEAN_RANK_MAX = 2.5
E5_POSITION_MAX = 2
G1_ID_RATIO_MAX = 0.333
G1_ID_ABS_DR_CAP = 0.35
DP_WIDTH_FRACTION = 0.2

# Capped grid: explicit cells (not full Cartesian product).
E5_CELL_SPECS: List[Dict[str, Any]] = [
    # T0 — complementary synthetic (K=2)
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "partial", "dp_eps": None},
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "full", "dp_eps": None},
    {"dgp": "two_client_complementary", "tier": "T0", "K_clients": 2, "policy_overlap": "partial", "dp_eps": 2.0},
    # T1 — semi-synthetic credit (K=2)
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "partial", "dp_eps": None},
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "full", "dp_eps": None},
    {"dgp": "semi_synthetic_credit", "tier": "T1", "K_clients": 2, "policy_overlap": "partial", "dp_eps": 2.0},
    # T2 — ACS multistate (K in {5, 10})
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "partial", "dp_eps": None},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 10, "policy_overlap": "partial", "dp_eps": None},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "full", "dp_eps": None},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 10, "policy_overlap": "full", "dp_eps": None},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 5, "policy_overlap": "partial", "dp_eps": 2.0},
    {"dgp": "acs_income_multistate", "tier": "T2", "K_clients": 10, "policy_overlap": "partial", "dp_eps": 2.0},
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SMOKE_SEEDS = [42, 43]
SMOKE_CELL_SPECS = [
    c for c in E5_CELL_SPECS
    if c["tier"] in ("T0", "T1") and c["policy_overlap"] == "partial" and c["dp_eps"] is None
]


def _json_sanitize(obj: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _cell_key(spec: Dict[str, Any], seed: int) -> Tuple[Any, ...]:
    return (
        seed,
        spec["dgp"],
        spec["policy_overlap"],
        spec["K_clients"],
        spec["dp_eps"],
    )


def _per_cell_ranks(rows: pd.DataFrame) -> pd.DataFrame:
    ranked: List[Dict[str, Any]] = []
    group_cols = ["seed", "dgp", "policy_overlap", "K_clients", "dp_eps"]
    for key, grp in rows.groupby(group_cols, dropna=False):
        sub = grp.sort_values("abs_bias").reset_index(drop=True)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            ranked.append(
                {
                    "seed": key[0],
                    "dgp": key[1],
                    "policy_overlap": key[2],
                    "K_clients": key[3],
                    "dp_eps": key[4] if pd.notna(key[4]) else None,
                    "method": row["method"],
                    "abs_bias": row["abs_bias"],
                    "cell_rank": rank,
                }
            )
    return pd.DataFrame(ranked)


def _aggregate_rank_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    agg = rows.groupby("method")["abs_bias"].mean().sort_values()
    summary: Dict[str, Any] = {}
    for rank, (method, mean_bias) in enumerate(agg.items(), start=1):
        summary[method] = {"mean_abs_bias": float(mean_bias), "mean_rank": float(rank)}
    return summary


def _primary_mask(df: pd.DataFrame) -> pd.Series:
    return df["dp_eps"].isna() | df["dp_eps"].isnull()


def _evaluate_g1_id(rows: pd.DataFrame) -> Dict[str, Any]:
    """G1-ID on complementary DGP, partial overlap, non-DP cells."""
    mask = (
        (rows["dgp"] == "two_client_complementary")
        & (rows["policy_overlap"] == "partial")
        & (_primary_mask(rows))
    )
    sub = rows[mask]
    reasons: List[str] = []
    passed = True
    if sub.empty:
        return {"pass": False, "reasons": ["No G1-ID cells found"], "cells": []}

    cells: List[Dict[str, Any]] = []
    for key, grp in sub.groupby(["seed"]):
        dr = grp[grp["method"] == "federated_dr"]["abs_bias"].iloc[0]
        avg = grp[grp["method"] == "naive_fedavg"]["abs_bias"].iloc[0]
        ratio = dr / max(avg, 1e-9)
        cell_pass = bool(ratio <= G1_ID_RATIO_MAX and dr <= G1_ID_ABS_DR_CAP)
        cells.append(
            {
                "seed": int(key[0]) if isinstance(key, tuple) else int(key),
                "bias_federated_dr": float(dr),
                "bias_naive_fedavg": float(avg),
                "ratio": float(ratio),
                "pass": cell_pass,
            }
        )
        if not cell_pass:
            passed = False

    n_pass = sum(1 for c in cells if c["pass"])
    reasons.append(f"G1-ID: {n_pass}/{len(cells)} complementary partial non-DP cells pass")
    for c in cells:
        status = "PASS" if c["pass"] else "FAIL"
        reasons.append(
            f"  seed={c['seed']}: |bias_DR|={c['bias_federated_dr']:.4f}, "
            f"ratio={c['ratio']:.3f} [{status}]"
        )
    return {"pass": bool(passed), "reasons": reasons, "cells": cells}


def _evaluate_dp_cells(rows: pd.DataFrame) -> Dict[str, Any]:
    """Informational DP width check (not a hard E5 fail unless all unusable)."""
    dp_rows = rows[~_primary_mask(rows)]
    info: List[Dict[str, Any]] = []
    for key, grp in dp_rows.groupby(["seed", "dgp", "policy_overlap", "K_clients", "dp_eps"]):
        dr_row = grp[grp["method"] == "federated_dr"]
        if dr_row.empty:
            continue
        row = dr_row.iloc[0]
        oracle = float(row["oracle"])
        width = row.get("dp_interval_width")
        oracle_range = abs(oracle) + 1e-6
        useful = width is not None and np.isfinite(width) and width <= DP_WIDTH_FRACTION * oracle_range
        info.append(
            {
                "seed": key[0],
                "dgp": key[1],
                "policy_overlap": key[2],
                "K_clients": key[3],
                "dp_eps": key[4],
                "interval_width": None if width is None or (isinstance(width, float) and np.isnan(width)) else float(width),
                "oracle": oracle,
                "width_fraction_of_oracle": None
                if width is None or not np.isfinite(width)
                else float(width / oracle_range),
                "decision_useful": bool(useful),
            }
        )
    n_useful = sum(1 for x in info if x["decision_useful"])
    return {
        "informational": True,
        "n_dp_cells": len(info),
        "n_decision_useful": n_useful,
        "cells": info,
    }


def _evaluate_e5_gate(
    rows: pd.DataFrame,
    rank_summary: Dict[str, Any],
    cell_ranks: pd.DataFrame,
    g1_id: Dict[str, Any],
    dp_info: Dict[str, Any],
) -> Dict[str, Any]:
    reasons: List[str] = []
    passed = True

    finite_ok = rows["estimate"].apply(np.isfinite).all()
    if finite_ok:
        reasons.append("All methods emit finite estimates (no crashes)")
    else:
        passed = False
        reasons.append("FAIL: non-finite estimates detected")

    primary_ranks = cell_ranks[_primary_mask(cell_ranks)]
    dr_primary = primary_ranks[primary_ranks["method"] == "federated_dr"]
    dr_mean_cell_rank = float(dr_primary["cell_rank"].mean()) if not dr_primary.empty else float("nan")
    if np.isfinite(dr_mean_cell_rank) and dr_mean_cell_rank <= E5_MEAN_RANK_MAX:
        reasons.append(
            f"federated_dr mean cell rank (primary non-DP) {dr_mean_cell_rank:.2f} <= {E5_MEAN_RANK_MAX}"
        )
    else:
        passed = False
        reasons.append(
            f"FAIL: federated_dr mean cell rank {dr_mean_cell_rank:.2f} > {E5_MEAN_RANK_MAX}"
        )

    dr_info = rank_summary.get("federated_dr", {})
    dr_position = int(dr_info.get("mean_rank", 999))
    if dr_position <= E5_POSITION_MAX:
        reasons.append(
            f"federated_dr aggregate position {dr_position} <= {E5_POSITION_MAX} "
            f"(mean |bias|={dr_info.get('mean_abs_bias', float('nan')):.4f})"
        )
    else:
        passed = False
        reasons.append(f"FAIL: federated_dr aggregate position {dr_position} > {E5_POSITION_MAX}")

    if g1_id.get("pass"):
        reasons.append("G1-ID holds on complementary partial non-DP cells")
    else:
        passed = False
        reasons.extend(g1_id.get("reasons", ["G1-ID FAIL"]))

    reasons.append(
        f"DP cells (informational): {dp_info.get('n_decision_useful', 0)}/"
        f"{dp_info.get('n_dp_cells', 0)} decision-useful at eps=2"
    )

    return {
        "pass": bool(passed),
        "reasons": reasons,
        "federated_dr_mean_cell_rank_primary": dr_mean_cell_rank,
        "federated_dr_aggregate_position": dr_position,
        "g1_id": g1_id,
        "dp_info": dp_info,
        "gate_thresholds": {
            "mean_cell_rank_max": E5_MEAN_RANK_MAX,
            "aggregate_position_max": E5_POSITION_MAX,
            "g1_id_ratio_max": G1_ID_RATIO_MAX,
            "g1_id_abs_dr_cap": G1_ID_ABS_DR_CAP,
        },
    }


def write_e5_report(
    path: Path,
    rows: pd.DataFrame,
    rank_summary: Dict[str, Any],
    cell_ranks: pd.DataFrame,
    gate: Dict[str, Any],
    seeds: List[int],
    wall_sec: float,
    t2_source: str,
) -> None:
    lines = [
        "# Idea 1 E5 Matrix Report",
        "",
        f"**Overall PASS:** `{gate['pass']}`",
        f"**Wall time:** {wall_sec:.1f}s",
        f"**Seeds:** {seeds}",
        f"**T2 data source:** {t2_source}",
        f"**Methods:** {len(E5_METHODS)} pilot-active (oracle excluded from ranks)",
        "",
        "## E5 gate (stricter than E3)",
        "",
    ]
    for r in gate.get("reasons", []):
        lines.append(f"- {r}")

    lines.extend(
        [
            "",
            f"- federated_dr mean primary cell rank: **{gate.get('federated_dr_mean_cell_rank_primary', float('nan')):.2f}**",
            f"- federated_dr aggregate position: **{gate.get('federated_dr_aggregate_position', '?')}**",
            "",
            "## Mean |bias| ranks (all cells, excl. oracle)",
            "",
            "| Rank | Method | Mean |bias| |",
            "|------|--------|-------------|",
        ]
    )
    for method, info in sorted(rank_summary.items(), key=lambda kv: kv[1]["mean_rank"]):
        lines.append(
            f"| {info['mean_rank']:.0f} | {method} | {info['mean_abs_bias']:.4f} |"
        )

    lines.extend(["", "## G1-ID complementary cells", ""])
    for c in gate.get("g1_id", {}).get("cells", []):
        lines.append(
            f"- seed={c['seed']}: |bias_DR|={c['bias_federated_dr']:.4f}, "
            f"ratio={c['ratio']:.3f}, pass={c['pass']}"
        )

    lines.extend(["", "## DP cells (informational)", ""])
    for c in gate.get("dp_info", {}).get("cells", []):
        lines.append(
            f"- {c['dgp']} seed={c['seed']} K={c['K_clients']} overlap={c['policy_overlap']}: "
            f"width={c['interval_width']}, useful={c['decision_useful']}"
        )

    lines.extend(["", "## Cell manifest", ""])
    for spec in E5_CELL_SPECS:
        lines.append(
            f"- {spec['tier']} {spec['dgp']} K={spec['K_clients']} "
            f"overlap={spec['policy_overlap']} dp={spec['dp_eps']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
            dp_eps = spec["dp_eps"]
            estimates = run_protocol_estimators(
                data, seed=seed, dp_eps=dp_eps, n_rounds=n_rounds
            )
            dr_interval = None
            dr_width = None
            if dp_eps is not None:
                dr_res = estimate_federated_dr(data, seed=seed, dp_eps=dp_eps)
                dr_interval = dr_res.interval
                dr_width = interval_width(dr_interval)

            for method in E5_METHODS:
                est = estimates[method]
                rows.append(
                    {
                        "seed": seed,
                        "dgp": spec["dgp"],
                        "tier": spec["tier"],
                        "policy_overlap": spec["policy_overlap"],
                        "K_clients": spec["K_clients"],
                        "dp_eps": dp_eps,
                        "method": method,
                        "estimate": est,
                        "oracle": oracle,
                        "bias": est - oracle,
                        "abs_bias": bias_vs_oracle(est, oracle),
                        "dp_interval_width": dr_width if method == "federated_dr" else None,
                    }
                )

    return pd.DataFrame(rows), t2_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Idea 1 E5 capped stress matrix")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=2000, help="samples per client")
    parser.add_argument("--n-rounds", type=int, default=2, dest="n_rounds")
    parser.add_argument("--out-dir", type=str, default="results/idea1_e5_local")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short local subset: 2 seeds x T0+T1 partial non-DP",
    )
    parser.add_argument(
        "--regate-from-csv",
        type=str,
        default=None,
        help="Recompute gate/report from existing results.csv (skip matrix run)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cell_specs = SMOKE_CELL_SPECS if args.smoke else E5_CELL_SPECS
    if args.smoke:
        seeds = [s for s in seeds if s in SMOKE_SEEDS][:2] or SMOKE_SEEDS

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.regate_from_csv:
        df = pd.read_csv(args.regate_from_csv)
        t2_source = "from_csv"
        wall_sec = 0.0
    else:
        t0 = time.perf_counter()
        df, t2_source = run_matrix(seeds, cell_specs, args.n, args.n_rounds)
        wall_sec = time.perf_counter() - t0
        df.to_csv(out_dir / "results.csv", index=False)

    rank_summary = _aggregate_rank_summary(df)
    cell_ranks = _per_cell_ranks(df)
    g1_id = _evaluate_g1_id(df)
    dp_info = _evaluate_dp_cells(df)
    gate = _evaluate_e5_gate(df, rank_summary, cell_ranks, g1_id, dp_info)
    gate["phase"] = "E5"
    gate["seeds"] = seeds
    gate["n_cells"] = int(
        df.groupby(["seed", "dgp", "policy_overlap", "K_clients", "dp_eps"], dropna=False).ngroups
    )
    gate["wall_sec"] = wall_sec
    gate["t2_data_source"] = t2_source if t2_source != "from_csv" else gate.get("t2_data_source", t2_source)
    gate["smoke_mode"] = args.smoke

    rank_payload = {
        "methods": rank_summary,
        "federated_dr_mean_cell_rank_primary": gate["federated_dr_mean_cell_rank_primary"],
        "federated_dr_aggregate_position": gate["federated_dr_aggregate_position"],
        "n_cells": gate["n_cells"],
        "seeds": seeds,
        "cell_specs": cell_specs,
    }
    write_json(out_dir / "rank_summary.json", _json_sanitize(rank_payload))
    write_json(out_dir / "e5_gate.json", _json_sanitize(gate))

    write_e5_report(
        out_dir / "E5_MATRIX_REPORT.md",
        df,
        rank_summary,
        cell_ranks,
        gate,
        seeds,
        wall_sec,
        t2_source,
    )

    print(f"E5 matrix PASS={gate['pass']} smoke={args.smoke} wall={wall_sec:.1f}s")
    print(f"federated_dr mean primary cell rank={gate['federated_dr_mean_cell_rank_primary']:.2f}")
    print(f"T2 source={t2_source}")
    print(f"Wrote {out_dir}")
    return 0 if gate["pass"] or args.smoke else 1


if __name__ == "__main__":
    raise SystemExit(main())
