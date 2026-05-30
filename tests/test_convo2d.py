"""Unit tests for Convo2dModel dataset and inference helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from Convo2dModel import (
    BINARY_LABELS,
    collect_labeled_pool,
    ensure_binary_split,
    predict_label,
    resolve_data_dir,
    stratified_train_val_split,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def _make_dataset(root: Path) -> None:
    for split, mapping in (
        ("Train", {"CROP_ADULT": "old", "CROP_CHILD": "teen"}),
        ("Validate", {"CROP_ADULT": "old", "CROP_CHILD": "teen"}),
    ):
        for folder, _label in mapping.items():
            for idx in range(3):
                _write_image(
                    root / split / folder / f"{split}_{folder}_{idx}.png",
                    (idx * 20, 10, 10) if folder == "CROP_ADULT" else (10, idx * 20, 10),
                )
    _write_image(root / "Test" / "holdout.png", (5, 5, 5))


def test_resolve_data_dir_prefers_existing_train(tmp_path: Path) -> None:
    empty = tmp_path / "DentalImages3"
    empty.mkdir()
    real = tmp_path / "DentalImages"
    _make_dataset(real)
    assert resolve_data_dir(tmp_path) == real


def test_collect_labeled_pool_dedupes_validate(tmp_path: Path) -> None:
    root = tmp_path / "DentalImages3"
    _make_dataset(root)
    pool = collect_labeled_pool(root)
    assert len(pool) == 6
    labels = [label for _, label in pool]
    assert labels.count("old") == 3
    assert labels.count("teen") == 3


def test_stratified_train_val_split_preserves_classes(tmp_path: Path) -> None:
    root = tmp_path / "DentalImages3"
    _make_dataset(root)
    pool = collect_labeled_pool(root)
    train_records, val_records = stratified_train_val_split(pool, val_fraction=0.34)
    assert len(train_records) + len(val_records) == len(pool)
    train_labels = {label for _, label in train_records}
    val_labels = {label for _, label in val_records}
    assert train_labels == set(BINARY_LABELS)
    assert val_labels == set(BINARY_LABELS)


def test_ensure_binary_split_builds_two_class_folders(tmp_path: Path) -> None:
    root = tmp_path / "DentalImages3"
    _make_dataset(root)
    pool = collect_labeled_pool(root)
    train_records, val_records = stratified_train_val_split(pool)
    binary_root = ensure_binary_split("pytest_train", train_records)
    assert (binary_root / "old").is_dir()
    assert (binary_root / "teen").is_dir()
    assert len(list((binary_root / "old").iterdir())) == len(
        [1 for _, label in train_records if label == "old"]
    )


def test_predict_label_returns_both_class_probs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indices_path = tmp_path / "class_indices.json"
    indices_path.write_text(json.dumps({"old": 0, "teen": 1}))
    monkeypatch.setattr("Convo2dModel.CLASS_INDICES_PATH", indices_path)

    class DummyModel:
        def predict(self, batch, verbose=0):
            return np.array([[0.25]], dtype=np.float32)

    pil = Image.new("RGB", (32, 32), (128, 128, 128))
    label, confidence, prob_teen, prob_old = predict_label(DummyModel(), pil)
    assert label == "old"
    assert prob_teen == pytest.approx(0.25)
    assert prob_old == pytest.approx(0.75)
    assert confidence == pytest.approx(0.75)
