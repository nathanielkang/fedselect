#!/usr/bin/env python3
"""Quick T2-only ablation for federated_dr T2 upgrade (seeds 42-44)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.idea1_censored.dgp import make_dgp
from src.idea1_censored.estimators import (
    _estimate_federated_dr_core,
    estimate_chen_icml2025_pooled_dr,
    estimate_fed_aipw_xiong,
    estimate_federated_dr,
)

SEEDS = [42, 43, 44]
N_PER = 2000


def mean_t2_bias(fn, label: str) -> float:
    biases = []
    for seed in SEEDS:
        for overlap in ("partial", "full"):
            data = make_dgp(
                "acs_income_multistate",
                seed=seed,
                n_per_client=N_PER,
                policy_overlap=overlap,
                n_clients=5,
            )
            oracle = data.oracle_risk()
            est = fn(data, seed)
            biases.append(abs(est - oracle))
    return float(np.mean(biases))


def main() -> int:
    baseline_dr = mean_t2_bias(
        lambda d, s: estimate_federated_dr(d, seed=s).estimate, "default"
    )
    xiong = mean_t2_bias(
        lambda d, s: estimate_fed_aipw_xiong(d, seed=s).estimate, "xiong"
    )
    chen = mean_t2_bias(
        lambda d, s: estimate_chen_icml2025_pooled_dr(d, seed=s).estimate, "chen"
    )

    print(f"Baselines T2 mean |bias|: default_dr={baseline_dr:.4f} xiong={xiong:.4f} chen={chen:.4f}")
    print()

    grid = []
    for clip, prox, blend in itertools.product(
        [0.02, 0.03, 0.05],
        [0, 1, 2, 3],
        [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
    ):
        def _run(data, seed, clip=clip, prox=prox, blend=blend):
            return _estimate_federated_dr_core(
                data,
                seed=seed,
                propensity_clip=clip,
                prox_rounds=prox,
                local_blend=blend,
                use_overlap_weights=True,
            ).estimate

        t2 = mean_t2_bias(_run, f"clip={clip} prox={prox} blend={blend}")
        grid.append((t2, clip, prox, blend))

    grid.sort(key=lambda x: x[0])
    print("Top 10 T2 configs (mean |bias| over seeds 42-44 x overlap):")
    for t2, clip, prox, blend in grid[:10]:
        beat_x = "YES" if t2 < xiong else "no"
        beat_chen = "YES" if t2 <= chen else "no"
        print(
            f"  T2={t2:.4f} clip={clip} prox={prox} blend={blend} "
            f"beat_xiong={beat_x} beat_chen={beat_chen}"
        )

    best_t2, best_clip, best_prox, best_blend = grid[0]
    print()
    print(
        f"BEST: T2={best_t2:.4f} clip={best_clip} prox={best_prox} blend={best_blend}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
