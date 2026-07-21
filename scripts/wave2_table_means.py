"""Compute Wave 2 manuscript table means from the frozen E5 matrix.

Reads results/gcp_e5/results.csv and prints per-tier mean |bias| for the
primary (non-DP) rows used in tab:main, plus the T0 partial identification
gate summary used in tab:identification-gate. No numbers are hardcoded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
CSV = HERE / "results" / "gcp_e5" / "results.csv"

ORDER = [
    "federated_dr",
    "federated_ipw",
    "naive_fedavg",
    "fedprox",
    "local_ipw",
    "scaffold",
    "fedpu_adapted",
    "local_only",
]


def main() -> None:
    df = pd.read_csv(CSV)
    prim = df[df["dp_eps"].isna()].copy()

    piv = prim.pivot_table(
        index="method", columns="tier", values="abs_bias", aggfunc="mean"
    )
    piv["Aggregate"] = prim.groupby("method")["abs_bias"].mean()
    piv = piv.reindex(ORDER)
    print("=== tab:main mean |bias| (primary, non-DP) ===")
    print(piv.round(4).to_string())

    # T0 complementary, partial, non-DP identification gate (ratio<=1/3, |bias|<=0.35)
    t0 = prim[(prim["tier"] == "T0") & (prim["policy_overlap"] == "partial")]
    print("\n=== T0 partial identification gate (per seed) ===")
    for seed in sorted(t0["seed"].unique()):
        s = t0[t0["seed"] == seed]
        b_dr = float(s[s["method"] == "federated_dr"]["abs_bias"].iloc[0])
        b_avg = float(s[s["method"] == "naive_fedavg"]["abs_bias"].iloc[0])
        local = s[s["method"].str.startswith("local_")]["abs_bias"]
        b_local = float(local.max()) if len(local) else float("nan")
        ratio_ok = b_dr <= b_avg / 3.0 and b_dr <= b_local / 3.0
        abs_ok = b_dr <= 0.35
        print(
            f"seed {seed}: |bias_DR|={b_dr:.4f} avg={b_avg:.4f} "
            f"worstlocal={b_local:.4f} ratio_ok={ratio_ok} abs_ok={abs_ok} "
            f"PASS={ratio_ok and abs_ok}"
        )

    # DP epsilon=2 interval width summary (informational)
    dp = df[df["dp_eps"] == 2.0]
    dpw = dp[dp["method"].isin(["federated_dr", "federated_ipw"])]
    print("\n=== DP eps=2 mean interval width (informational) ===")
    print(dpw.groupby("method")["dp_interval_width"].mean().round(4).to_string())

    # per-overlap aggregate for fig4 / narrative
    print("\n=== mean |bias| by overlap regime (primary) ===")
    ov = prim.pivot_table(
        index="method", columns="policy_overlap", values="abs_bias", aggfunc="mean"
    ).reindex(ORDER)
    print(ov.round(4).to_string())


if __name__ == "__main__":
    main()
