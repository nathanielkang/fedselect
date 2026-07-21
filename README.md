# FedSelect

**FedSelect** estimates population counterfactual risk under **decision-selective labeling** in federated learning: each client reveals outcomes only for the units it selects, selection policies differ across clients, and raw labels are never pooled.

The estimator pairs a local selection (propensity) model with a shared outcome bridge built by leave-one-client-out aggregation of outcome coefficients, then aggregates doubly robust moments at the server (optional differential-privacy noise on the release).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

From this directory, put the package root on `PYTHONPATH`:

```bash
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path
# Unix
export PYTHONPATH="$PWD"
```

## Quick start (smoke)

```bash
python scripts/run_idea1_smoke.py --seed 42 --out-dir results/idea1_smoke
```

Exit code `0` means at least one data-generating process passed the identification-style kill gate. Outputs are written under `results/` (gitignored); re-run the scripts to reproduce metrics locally.

Optional private release of aggregated moments:

```bash
python scripts/run_idea1_smoke.py --seed 42 --dp-eps 1.0 --out-dir results/idea1_smoke_dp
```

## Main evaluation scripts

| Script | Role |
|--------|------|
| `scripts/run_idea1_smoke.py` | Fast local smoke / kill gate |
| `scripts/run_idea1_v2_baselines.py` | V2 baseline matrix (FedSelect, UCL pooled adaptation, Fed-AIPW, …) |
| `scripts/run_idea1_e5_matrix.py` | Larger multi-seed stress matrix |
| `scripts/run_idea1_ablation.py` | Ablation grid |

## Layout

| Path | Role |
|------|------|
| `src/idea1_censored/` | DGPs, FedSelect / baselines, metrics, protocol |
| `scripts/` | Smoke, baselines, pilot, and matrix CLIs |
| `requirements.txt` | Python dependencies |

## Methods (selection)

| Key | Description |
|-----|-------------|
| `federated_dr` | **FedSelect** — federated doubly robust estimator |
| `chen_icml2025_pooled_dr` | Pooled cross-fit DR ceiling (UCL-inspired; privacy waived) |
| `fed_aipw_xiong` | Federated AIPW with shared outcome bridge (Xiong-style) |
| `federated_ipw` / `local_ipw` | Inverse-propensity baselines |
| `naive_fedavg` / `fedprox` | Selection-blind federated optimizers |
| `fedpu_adapted` | Positive-unlabeled adaptation |

Ground-truth risk uses synthetic counterfactuals `Y(1)` that clients never observe.

## License

MIT — see `LICENSE`.

## Contact

Nathaniel Kang  
School of Computer Science and Engineering  
Kyungpook National University  
natekang@knu.ac.kr
