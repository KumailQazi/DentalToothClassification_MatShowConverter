import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_PROJECT_ROOT = Path(__file__).resolve().parent
_DATASET_DIR_NAMES = (
    "DentalImages3",
    "dental images 3",
    "Dental Images 3",
    "DentalImages",
    "DentalImages_Cropped",
    "DentalImagesupdated",
)
_SOURCE_TO_BINARY: dict[str, str] = {
    "CROP_ADULT": "old",
    "CROP_ADOLESCENT": "teen",
    "CROP_CHILD": "teen",
}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
BINARY_LABELS = ("old", "teen")
VAL_FRACTION = 0.2
RANDOM_SEED = 42


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Fix seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_data_dir(root: Path | None = None) -> Path:
    """Return the first existing dataset root that contains a Train/ split."""
    root = root or _PROJECT_ROOT
    for name in _DATASET_DIR_NAMES:
        candidate = root / name
        if (candidate / "Train").is_dir():
            return candidate
    return root / _DATASET_DIR_NAMES[0]


DATA_DIR = resolve_data_dir()
BINARY_CACHE_DIR = DATA_DIR / "_binary"
MODEL_PATH = _PROJECT_ROOT / "tooth_classifier.keras"
CLASS_INDICES_PATH = _PROJECT_ROOT / "class_indices.json"
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 8
EPOCHS = 5


def _data_dir(name: str) -> Path:
    path = DATA_DIR / name
    if not path.is_dir():
        tried = ", ".join(_DATASET_DIR_NAMES)
        raise FileNotFoundError(
            f"Expected dataset folder at {path}. "
            f"Tried names under {_PROJECT_ROOT}: {tried}. "
            "Each dataset root must contain Train/ and Test/."
        )
    return path


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            count += 1
    return count


def dataset_counts() -> dict[str, dict[str, int] | int]:
    """Return image counts per split (and per raw class folder when present)."""
    counts: dict[str, dict[str, int] | int] = {}
    for split in ("Train", "Validate", "Test"):
        split_dir = DATA_DIR / split
        if not split_dir.is_dir():
            counts[split] = 0
            continue
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if subdirs:
            counts[split] = {
                d.name: _count_images(d) for d in sorted(subdirs, key=lambda p: p.name)
            }
        else:
            counts[split] = _count_images(split_dir)
    counts["total"] = sum(
        v if isinstance(v, int) else sum(v.values()) for v in counts.values()
    )
    return counts


def _file_digest(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_from_raw_split(split_dir: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    if not split_dir.is_dir():
        return records
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _SOURCE_TO_BINARY.get(class_dir.name)
        if label is None:
            raise ValueError(
                f"Unknown class folder {class_dir.name!r} under {split_dir}. "
                f"Expected one of {sorted(_SOURCE_TO_BINARY)}."
            )
        for img in sorted(class_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in _IMAGE_EXTENSIONS:
                records.append((img.resolve(), label))
    return records


def collect_labeled_pool(data_dir: Path | None = None) -> list[tuple[Path, str]]:
    """
    Collect unique labeled images from Train (and Validate only for dedupe).

    The on-disk Validate/ folder is a duplicate of Train/ in this dataset; it is
    not used as a validation set. Images are deduplicated by file content hash.
    """
    data_dir = data_dir or DATA_DIR
    pool: list[tuple[Path, str]] = []
    for split_name in ("Train", "Validate"):
        pool.extend(_records_from_raw_split(data_dir / split_name))

    unique: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, label in pool:
        digest = _file_digest(path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((path, label))
    if not unique:
        raise FileNotFoundError(
            f"No labeled images found under {data_dir / 'Train'}. "
            "Expected CROP_ADULT, CROP_ADOLESCENT, and CROP_CHILD subfolders."
        )
    return unique


def stratified_train_val_split(
    pool: list[tuple[Path, str]],
    *,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Split labeled pool into stratified train and validation lists."""
    paths = [path for path, _ in pool]
    labels = [label for _, label in pool]
    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths,
            labels,
            test_size=val_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        # Too few samples per class for sklearn stratify — split per class.
        by_label: dict[str, list[Path]] = {label: [] for label in BINARY_LABELS}
        for path, label in pool:
            by_label[label].append(path)
        train_paths, val_paths = [], []
        train_labels, val_labels = [], []
        rng = random.Random(seed)
        for label, label_paths in by_label.items():
            rng.shuffle(label_paths)
            if len(label_paths) < 2:
                train_paths.extend(label_paths)
                train_labels.extend([label] * len(label_paths))
                continue
            cut = max(1, int(round(len(label_paths) * (1 - val_fraction))))
            cut = min(cut, len(label_paths) - 1)
            train_paths.extend(label_paths[:cut])
            train_labels.extend([label] * cut)
            val_paths.extend(label_paths[cut:])
            val_labels.extend([label] * (len(label_paths) - cut))
    train_records = list(zip(train_paths, train_labels, strict=True))
    val_records = list(zip(val_paths, val_labels, strict=True))
    return train_records, val_records


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dest)


def ensure_binary_split(
    split: str,
    records: list[tuple[Path, str]] | None = None,
) -> Path:
    """Build a two-class folder layout (old, teen) for Keras flow_from_directory."""
    if records is None:
        records = _records_from_raw_split(_data_dir(split))

    binary_root = BINARY_CACHE_DIR / split
    for label in BINARY_LABELS:
        label_dir = binary_root / label
        if label_dir.exists():
            for existing in label_dir.iterdir():
                if existing.is_symlink() or existing.is_file():
                    existing.unlink()
        else:
            label_dir.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, set[str]] = {label: set() for label in BINARY_LABELS}
    for src, label in records:
        dest_name = src.name
        if dest_name in used_names[label]:
            dest_name = f"{src.stem}_{_file_digest(src)[:8]}{src.suffix}"
        used_names[label].add(dest_name)
        _link_or_copy(src, binary_root / label / dest_name)

    for label in BINARY_LABELS:
        if not any((binary_root / label).iterdir()):
            raise FileNotFoundError(
                f"No images linked for {label!r} in {binary_root}. "
                "Check label mapping and input records."
            )
    return binary_root


def _class_weight_dict(
    records: list[tuple[Path, str]], class_indices: dict[str, int]
) -> dict[int, float]:
    labels = [class_indices[label] for _, label in records]
    classes = np.array(sorted(set(labels)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights, strict=True)}


def build_model(trainable_base: bool = False) -> tf.keras.Model:
    """MobileNetV2 transfer-learning head for binary old vs teen."""
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="tooth_mobilenetv2")


def _train_val_generators(
    train_dir: Path,
    val_dir: Path,
    *,
    batch_size: int,
) -> tuple[ImageDataGenerator, ImageDataGenerator, tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_dataset = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
    )
    validation_dataset = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return train_datagen, val_datagen, train_dataset, validation_dataset


def log_validation_report(
    model: tf.keras.Model,
    validation_dataset: tf.keras.utils.Sequence,
    *,
    class_indices: dict[str, int] | None = None,
) -> None:
    """Print confusion matrix and sklearn classification report on validation data."""
    class_indices = class_indices or {"old": 0, "teen": 1}
    index_to_label = {int(v): k for k, v in class_indices.items()}
    validation_dataset.reset()
    y_prob = model.predict(validation_dataset, verbose=0).reshape(-1)
    validation_dataset.reset()
    y_true = np.concatenate([labels for _, labels in validation_dataset]).astype(int)

    teen_index = class_indices.get("teen", 1)
    y_pred = np.where(y_prob >= 0.5, teen_index, 1 - teen_index)
    labels_order = [index_to_label[i] for i in sorted(index_to_label)]
    label_ids = [class_indices[name] for name in labels_order]
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=label_ids))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels_order,
            zero_division=0,
        )
    )


def predict_label(
    model: tf.keras.Model,
    pil_image,
) -> tuple[str, float, float, float]:
    """Return (label, confidence, P(teen), P(old)) for a PIL image."""
    if CLASS_INDICES_PATH.exists():
        class_indices = json.loads(CLASS_INDICES_PATH.read_text())
    else:
        class_indices = {"old": 0, "teen": 1}
    index_to_label = {int(v): k for k, v in class_indices.items()}
    teen_index = class_indices.get("teen", 1)

    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = image.img_to_array(img)
    batch = np.expand_dims(arr, axis=0) / 255.0
    prob_teen = float(model.predict(batch, verbose=0)[0][0])
    prob_old = 1.0 - prob_teen
    pred_index = teen_index if prob_teen >= 0.5 else (1 - teen_index)
    label = index_to_label.get(pred_index, "unknown")
    confidence = prob_teen if pred_index == teen_index else prob_old
    return label, confidence, prob_teen, prob_old


def tooth_classification(
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_path: Path = MODEL_PATH,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    set_random_seeds()
    pool = collect_labeled_pool()
    train_records, val_records = stratified_train_val_split(pool)
    train_dir = ensure_binary_split("train", train_records)
    val_dir = ensure_binary_split("val", val_records)

    _, _, train_dataset, validation_dataset = _train_val_generators(
        train_dir, val_dir, batch_size=batch_size
    )
    class_weight = _class_weight_dict(train_records, train_dataset.class_indices)

    model = build_model(trainable_base=False)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=1e-4),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    run_dir = _PROJECT_ROOT / "logs" / "fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        ModelCheckpoint(
            str(save_path),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        ),
        TensorBoard(log_dir=str(run_dir)),
    ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    CLASS_INDICES_PATH.write_text(
        json.dumps(train_dataset.class_indices, indent=2)
    )
    print("Resolved dataset:", DATA_DIR)
    print("Labeled pool (deduped):", len(pool))
    print("Stratified split — train:", len(train_records), "val:", len(val_records))
    print("Class indices:", train_dataset.class_indices)
    print("Class weights:", class_weight)
    log_validation_report(model, validation_dataset, class_indices=train_dataset.class_indices)
    return model, history


def predict_test_images(
    model: tf.keras.Model | None = None,
    *,
    show_plots: bool = False,
) -> None:
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No saved model at {MODEL_PATH}. Run training first."
            )
        model = tf.keras.models.load_model(MODEL_PATH)

    test_dir = _data_dir("Test")
    for img_path in sorted(test_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        if show_plots:
            plt.imshow(img)
            plt.title(img_path.name)
            plt.show()

        label, confidence, prob_teen, prob_old = predict_label(model, img)
        print(
            f"{img_path.name}: {label} "
            f"(confidence={confidence:.3f}, P(teen)={prob_teen:.3f}, P(old)={prob_old:.3f})"
        )


# Notebook compatibility
toothClassification = tooth_classification


if __name__ == "__main__":
    trained, hist = tooth_classification()
    last = hist.history
    print(
        f"Final epoch — loss: {last['loss'][-1]:.4f}, "
        f"accuracy: {last['accuracy'][-1]:.4f}, "
        f"val_auc: {last['val_auc'][-1]:.4f}, "
        f"val_loss: {last['val_loss'][-1]:.4f}"
    )
    predict_test_images(trained, show_plots=False)
