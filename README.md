# CP-MedShift

Conformal prediction for medical vision-language models (VLMs) under distribution shift: **label shift**, **device shift** (image corruptions) and **institution shift** (external test sets). It covers two scenarios:

- **Offline:** prediction sets on a whole test set, calibrated with a few labelled images (16 per class).
- **Online:** a stream of images that changes at an unknown point. A conformal test martingale detects the change, and online prediction sets adapt to the new class mix.

All experiments run on precomputed image embeddings and text prototypes from the VLMs (CONVIRT, CONCH, FLAIR), so no image model is needed.

## Contents

```
CP-MedShift/
├── conformal/                     # methods
│   ├── conformal_prediction.py    # ICP, FCP (full conformal adaptation, SS-Text), weighted FCP, Mondrian
│   ├── label_shift.py             # label weights, BBSE (soft confusion matrix), EMA estimate q̂_t of the class mix
│   ├── online.py                  # streams, conformal p-values, Simple Jumper martingale, w_online / w_oracle weights
│   └── metrics.py                 # coverage, class set size, UCV / CCV, moving-window metrics of streams
├── utils/
│   ├── embedding_datasets.py      # loading of the embeddings, class names, external test sets
│   ├── data_sampling.py           # calibration / stream sampling with a given class mix
│   ├── distribution_shift.py      # Dirichlet draws at a target KL, KL / L1, domain-classifier AUC
│   ├── linear_classifier.py       # zero-shot softmax scores
│   ├── progress.py                # progress printing
│   └── results_cache.py           # reuse of saved draws
├── notebooks/
│   ├── offline_shifts.ipynb       # offline experiments  -> results/offline_shifts_testmix_100draws.csv
│   ├── online_shifts.ipynb        # online experiments   -> results/online_streams.csv (+ alarm tables, curves)
│   └── make_paper_plots.ipynb     # paper figures from the CSV files -> results/figures/paper/
└── local_data -> ../local_data    # embeddings (not in git)
```

## Datasets

| Modality | VLM | Datasets (C = number of classes) | External sets (institution shift) |
|---|---|---|---|
| Radiology | CONVIRT | CheXpert5x200 (5), nihlt (19), covid (4) | – |
| Histology | CONCH | NCT (9), SICAPv2 (4), Skin (16) | ZURICH, GERTYCH (both for SICAPv2) |
| Ophthalmology | FLAIR | MESSIDOR (5), FIVES (4), MMAC (5) | DeepDRiD (for MESSIDOR) |

The expected layout of `local_data/embeddings/` is:
- `<dataset>_train_<vlm>.npz` and `<dataset>_test_<vlm>.npz`: embeddings, labels, text prototypes and logit scale;
- `<external>_test_<vlm>.npz`: the external test sets;
- `blur_L1` … `blur_L5` and `noise_L1` … `noise_L5`: the same files for the corrupted test splits (device shift).

## Methods

- **Offline:** ICP, FCP, FCP + Mondrian (class-wise), FCP + w_label (weights q̂_test / q_cal estimated by BBSE), and FCP + w_oracle (true weights).
- **Online martingales** (alarm at M_t ≥ 1/α = 100):
  - *unweighted*;
  - *w_online*: weights from the EMA estimate q̂_t of the class mix (λ = 1/50); does not use the change point;
  - *w_oracle*: true labels and true class mix, knows the change point.
- **Online prediction sets:** FCP, FCP + w_online (label weights q̂_t / q_cal), and FCP + w_oracle (q*_t / q_cal).

## Usage

From `notebooks/`, run them in this order:
1. `offline_shifts.ipynb`
2. `online_shifts.ipynb`
3. `make_paper_plots.ipynb`

The two experiment notebooks save after every case and skip what is already in their results CSV. FCP runs on the GPU when one is available (`cuda`, then `mps`); set `FCA_DEVICE` to override.
