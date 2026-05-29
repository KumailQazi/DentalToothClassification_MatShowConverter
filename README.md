# Dental Tooth Classification

Binary classification of dental X-ray images (**old** vs **teen**) using a TensorFlow/Keras model (MobileNetV2 transfer learning).

> **Disclaimer:** This is a research / teaching demo only. It is not validated for clinical diagnosis or treatment decisions.

## Setup

Requires **Python 3.10–3.12** (TensorFlow is not available on 3.13+).

```bash
python3.11 -m venv .venv          # or python3.10 / python3.12
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset layout

Place images under `DentalImages3/` at the project root (included locally, not in git). The training script also checks common aliases: `dental images 3`, `Dental Images 3`, `DentalImages`, etc.

```
DentalImages3/
  Train/
    CROP_ADOLESCENT/    # mapped → teen
    CROP_ADULT/         # mapped → old
    CROP_CHILD/         # mapped → teen
  Validate/             # optional; duplicate of Train in this repo — not used as val set
    (same three folders)
  Test/
    *.jpg               # flat folder, unlabeled holdout images
```

### Validation split (programmatic)

`Convo2dModel.py` does **not** use the on-disk `Validate/` folder for metrics. That folder is byte-identical to `Train/` in the current dataset. Instead:

1. Collect unique labeled images from `Train/` + `Validate/` (content-hash dedupe).
2. Apply an **~80/20 stratified train/val split** (scikit-learn, `random_state=42`).
3. Cache symlinks under `DentalImages3/_binary/train/` and `.../val/` for Keras.

Re-count after changes:

```bash
python -c "from Convo2dModel import dataset_counts, collect_labeled_pool; print(dataset_counts()); print('deduped labeled:', len(collect_labeled_pool()))"
```

### Class imbalance

Teen (adolescent + child) dominates the labeled pool. Training uses **balanced `class_weight`**, and reports **AUC, precision, recall**, plus a validation **confusion matrix** and **classification report** after `fit`.

## Train and evaluate

```bash
python Convo2dModel.py
```

Outputs:

- `tooth_classifier.keras` — best checkpoint by `val_auc`
- `class_indices.json` — label → index map from training
- `logs/fit/<timestamp>/` — TensorBoard event files

TensorBoard:

```bash
tensorboard --logdir logs/fit
```

### Metrics expectations

On this small, imbalanced set (~59 deduped labeled images), expect **high variance** between runs. Use **val_auc** and the printed classification report rather than accuracy alone. Holdout `Test/` images have no public labels in-repo.

## Tests

```bash
pytest tests/test_convo2d.py -q
```

## Streamlit demo

After training:

```bash
streamlit run app.py
```

Upload a cropped X-ray; the app shows **old** / **teen**, confidence, and **P(teen)** / **P(old)**.

## Notebooks

- `ToothClassification.ipynb` — original training notebook (`from Convo2dModel import toothClassification`)
- `Dental tooth classification MAT.ipynb` — **deprecated** legacy notebook (3-class Conv2D, `validation_data=train` bug). Use `python Convo2dModel.py` or `ToothClassification.ipynb` instead.

## Results / limitations

- Small dataset with class imbalance; metrics vary run-to-run.
- Mapping: **adolescent + child → teen**, **adult → old**; test images have no public labels in-repo.
- Model input is 200×200 RGB; expects cropped X-rays similar to training data.
- Not FDA-cleared, not peer-reviewed for clinical use—suitable for coursework and experimentation only.
