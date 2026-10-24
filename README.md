# BT4012 Kaggle competition: detecting illicit Bitcoin transactions

Student Name: Lai Hsuan Jung
Student ID: E0968110

Each Bitcoin transaction is described by 165 anonymised features and linked to other transactions
through payment flows. The task is to rank test transactions by how likely they are to be illicit, scored by ROC-AUC.

## Final model

Default LightGBM on the 165 raw features, blended (by rank) with the average of five directed GraphSAGE
graph neural networks: `0.6 x LightGBM rank + 0.4 x GNN rank`.

| Submission | Validation (mean of 3 time-based folds) | Public leaderboard |
|---|---|---|
| LightGBM alone (`sub01_baseline`) | 0.9808 | 0.94660 |
| LightGBM + 0.4 x GNN (`sub10_gnn5_blend_w04`, final) | 0.9770 | 0.95654 |

## How to reproduce the submission

1. Create the environment (Python 3.14 was used):

   ```
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Put the competition files in `data/`: `train.csv`, `test.csv`, `txs_edgelist.csv`,
   `sample_submission.csv`.
3. Open `e0968110.ipynb` from the repository root and run all cells (about 20 minutes). It writes
   `submissions/e0968110_final_submission.csv`.

The GNN part of the notebook is written to `gnn_step.py` and run as a separate Python process, because
PyTorch and LightGBM crash or freeze when trained in the same session on macOS. The notebook explains
this in its first cell.

## Repository contents

| Path | What it is |
|---|---|
| `e0968110.ipynb` | Final notebook: data, validation, both models, blend, submission |
| `notebooks/01_eda.ipynb` to `notebooks/17_class_weight.ipynb` | Experiment notebooks, in the order they were run |
| `experiment_log.md` | Every experiment and submission, with validation and leaderboard scores, and the AI-usage record |
| `requirements.txt` | Pinned package versions |
| `data/`, `submissions/`, `outputs/` | Not committed (competition data, submission files, saved GNN predictions) |

The experiment notebooks use paths relative to `notebooks/` (for example `../data/train.csv`), so open them from that folder.
