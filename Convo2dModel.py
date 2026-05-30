import json
import os
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop
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
# Map raw crop folders to binary labels (old vs teen).
_SOURCE_TO_BINARY: dict[str, str] = {
    "CROP_ADULT": "old",
    "CROP_ADOLESCENT": "teen",
    "CROP_CHILD": "teen",
}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


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
            "Each dataset root must contain Train/, Validate/, and Test/."
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


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dest)


def ensure_binary_split(split: str) -> Path:
    """Build Train/Validate layout with exactly two class folders: old, teen."""
    source = _data_dir(split)
    binary_root = BINARY_CACHE_DIR / split
    for label in ("old", "teen"):
        (binary_root / label).mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _SOURCE_TO_BINARY.get(class_dir.name)
        if label is None:
            raise ValueError(
                f"Unknown class folder {class_dir.name!r} under {source}. "
                f"Expected one of {sorted(_SOURCE_TO_BINARY)}."
            )
        dest_dir = binary_root / label
        for img in sorted(class_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in _IMAGE_EXTENSIONS:
                _link_or_copy(img, dest_dir / img.name)

    for label in ("old", "teen"):
        if not any((binary_root / label).iterdir()):
            raise FileNotFoundError(
                f"No images linked for {label!r} in {binary_root}. "
                "Check _SOURCE_TO_BINARY mapping and raw class folders."
            )
    return binary_root


def build_model() -> tf.keras.Model:
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(*IMAGE_SIZE, 3)
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def tooth_classification(
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_path: Path = MODEL_PATH,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    train_dir = ensure_binary_split("Train")
    val_dir = ensure_binary_split("Validate")

    train_gen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_dataset = train_gen.flow_from_directory(
        str(train_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
    validation_dataset = val_gen.flow_from_directory(
        str(val_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    model = build_model()
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=0.001),
        metrics=["accuracy"],
    )

    log_dir = _PROJECT_ROOT / "logs" / "fit"
    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True, monitor="val_accuracy"),
        TensorBoard(log_dir=str(log_dir)),
    ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    CLASS_INDICES_PATH.write_text(
        json.dumps(train_dataset.class_indices, indent=2)
    )
    print("Resolved dataset:", DATA_DIR)
    print("Class indices:", train_dataset.class_indices)
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
    if CLASS_INDICES_PATH.exists():
        class_indices = json.loads(CLASS_INDICES_PATH.read_text())
    else:
        class_indices = {"old": 0, "teen": 1}
    index_to_label = {int(v): k for k, v in class_indices.items()}
    teen_index = class_indices.get("teen", 1)

    for img_path in sorted(test_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        if show_plots:
            plt.imshow(img)
            plt.title(img_path.name)
            plt.show()

        arr = image.img_to_array(img)
        batch = np.expand_dims(arr, axis=0) / 255.0
        prob_teen = float(model.predict(batch, verbose=0)[0][0])
        pred_index = teen_index if prob_teen >= 0.5 else (1 - teen_index)
        label = index_to_label.get(pred_index, "unknown")
        confidence = prob_teen if pred_index == teen_index else 1.0 - prob_teen
        print(
            f"{img_path.name}: {label} "
            f"(confidence={confidence:.3f}, P(teen)={prob_teen:.3f})"
        )


# Notebook compatibility
toothClassification = tooth_classification


if __name__ == "__main__":
    trained, hist = tooth_classification()
    last = hist.history
    print(
        f"Final epoch — loss: {last['loss'][-1]:.4f}, "
        f"accuracy: {last['accuracy'][-1]:.4f}, "
        f"val_loss: {last['val_loss'][-1]:.4f}, "
        f"val_accuracy: {last['val_accuracy'][-1]:.4f}"
    )
    predict_test_images(trained, show_plots=False)
