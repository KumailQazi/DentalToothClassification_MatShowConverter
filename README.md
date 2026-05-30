# Dental Tooth Classification

Binary classification of dental X-ray images (**old** vs **teen**) using a TensorFlow/Keras Conv2D CNN.

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
  Validate/
    (same three folders)
  Test/
    *.jpg               # flat folder, unlabeled holdout images
```

### Image counts (current dataset)

| Split | CROP_ADOLESCENT | CROP_ADULT | CROP_CHILD | Flat Test |
|-------|-----------------|------------|------------|-----------|
| Train | 4 | 9 | 46 | — |
| Validate | 4 | 9 | 46 | — |
| Test | — | — | — | 29 |
| **Total** | | | | **147** |

`Convo2dModel.py` builds a cached binary layout at `DentalImages3/_binary/` (`old/` + `teen/`) via symlinks so Keras `class_mode="binary"` works with the three raw age folders.

Re-count after changes:

```bash
python -c "from Convo2dModel import dataset_counts; print(dataset_counts())"
```

## Train and evaluate

```bash
python Convo2dModel.py
```

Outputs:

- `tooth_classifier.keras` — best checkpoint by validation accuracy
- `class_indices.json` — label → index map from training
- `logs/fit/` — TensorBoard event files

TensorBoard:

```bash
tensorboard --logdir logs/fit
```

## Streamlit demo

After training:

```bash
streamlit run app.py
```

Upload a cropped X-ray; the app shows **old** / **teen**, confidence, and P(teen).

## Notebooks

- `ToothClassification.ipynb` — original training notebook (`from Convo2dModel import toothClassification`)
- `Dental tooth classification MAT.ipynb` — alternate Conv2D workflow

## Results / limitations

- Small dataset (~59 training images per epoch after binary merge: 50 teen + 9 old) with class imbalance; metrics vary run-to-run.
- Mapping: **adolescent + child → teen**, **adult → old**; test images have no public labels in-repo.
- Model input is 200×200 RGB; expects cropped X-rays similar to training data.
- Not FDA-cleared, not peer-reviewed for clinical use—suitable for coursework and experimentation only.
