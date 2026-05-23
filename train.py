"""
Train the sign language recognition model.

Usage:
    python train.py
"""

import os
import json
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, MODEL_DIR, MODEL_PATH, LABELS_PATH,
    SEQUENCE_LENGTH, NUM_FEATURES,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT,
)
from model import build_model, get_callbacks

MIN_SEQUENCES_PER_LABEL = 2


def load_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels_path = os.path.join(DATA_DIR, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"labels.json not found at {labels_path}. "
            "Run data_collector.py first."
        )

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels: dict[str, int] = json.load(f)

    per_label: dict[str, list[np.ndarray]] = {}
    for label_name, label_idx in labels.items():
        label_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(label_dir):
            print(f"  [WARN] No data directory for '{label_name}', skipping.")
            continue

        seqs = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        if not seqs:
            print(f"  [WARN] No sequences for '{label_name}', skipping.")
            continue

        valid = []
        for fname in seqs:
            path = os.path.join(label_dir, fname)
            seq = np.load(path)
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                valid.append(seq)
            else:
                print(f"  [WARN] Bad shape {seq.shape} in {path}, skipping.")

        per_label[label_name] = valid
        print(
            f"  Loaded {len(valid)}/{len(seqs)} valid sequences for "
            f"'{label_name}' (idx={label_idx})"
        )

    eligible = {
        name: seqs
        for name, seqs in per_label.items()
        if len(seqs) >= MIN_SEQUENCES_PER_LABEL
    }
    skipped = {
        name: len(seqs)
        for name, seqs in per_label.items()
        if 0 < len(seqs) < MIN_SEQUENCES_PER_LABEL
    }

    for name, count in skipped.items():
        needed = MIN_SEQUENCES_PER_LABEL - count
        print(
            f"  [WARN] '{name}' has only {count} valid sequence(s); "
            f"collect at least {needed} more. Skipping this label."
        )

    if not eligible:
        raise ValueError("No valid data found. Check your data directory.")

    if len(eligible) < 2:
        details = ", ".join(
            f"{name}: {len(seqs)} valid"
            for name, seqs in sorted(per_label.items(), key=lambda item: labels[item[0]])
        )
        raise ValueError(
            "Training needs at least 2 labels with 2 or more valid sequences each. "
            f"Current valid counts: {details}"
        )

    compact_labels = {
        name: new_idx
        for new_idx, name in enumerate(sorted(eligible, key=lambda item: labels[item]))
    }

    X, y = [], []
    for label_name, label_idx in compact_labels.items():
        for seq in eligible[label_name]:
            X.append(seq)
            y.append(label_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), compact_labels


def plot_history(history, save_path: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 420
    margin = 54
    gap = 46
    plot_w = (width - margin * 2 - gap) // 2
    plot_h = height - margin * 2
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def draw_panel(x0: int, title: str, train_key: str, val_key: str, y_min=None, y_max=None):
        train = [float(v) for v in history.history[train_key]]
        val = [float(v) for v in history.history[val_key]]
        values = train + val
        lo = min(values) if y_min is None else y_min
        hi = max(values) if y_max is None else y_max
        if hi == lo:
            hi = lo + 1.0
        pad = (hi - lo) * 0.08
        lo -= pad
        hi += pad

        y0 = margin
        x1 = x0 + plot_w
        y1 = y0 + plot_h
        draw.rectangle((x0, y0, x1, y1), outline=(30, 30, 30))
        draw.text((x0 + plot_w // 2 - 24, 18), title, fill=(0, 0, 0), font=font)

        ticks = 4
        for i in range(ticks + 1):
            y = y1 - int(plot_h * i / ticks)
            value = lo + (hi - lo) * i / ticks
            draw.line((x0, y, x1, y), fill=(232, 232, 232))
            draw.text((x0 - 42, y - 6), f"{value:.2f}", fill=(0, 0, 0), font=font)

        def points(series):
            if len(series) == 1:
                return [(x0, y1 - int((series[0] - lo) / (hi - lo) * plot_h))]
            return [
                (
                    x0 + int(i / (len(series) - 1) * plot_w),
                    y1 - int((v - lo) / (hi - lo) * plot_h),
                )
                for i, v in enumerate(series)
            ]

        train_pts = points(train)
        val_pts = points(val)
        if len(train_pts) > 1:
            draw.line(train_pts, fill=(31, 119, 180), width=3)
            draw.line(val_pts, fill=(255, 127, 14), width=3)
        for p in train_pts:
            draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=(31, 119, 180))
        for p in val_pts:
            draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=(255, 127, 14))

        lx = x1 - 105
        ly = y0 + 14
        draw.line((lx, ly, lx + 24, ly), fill=(31, 119, 180), width=3)
        draw.text((lx + 30, ly - 6), 'train', fill=(0, 0, 0), font=font)
        draw.line((lx, ly + 20, lx + 24, ly + 20), fill=(255, 127, 14), width=3)
        draw.text((lx + 30, ly + 14), 'val', fill=(0, 0, 0), font=font)

    draw_panel(margin, 'Accuracy', 'accuracy', 'val_accuracy', 0.0, 1.0)
    draw_panel(margin + plot_w + gap, 'Loss', 'loss', 'val_loss')
    img.save(save_path)
    print(f"Training plot saved to {save_path}")


def main() -> None:
    print("=== 한국수어 모델 학습 ===\n")

    print("데이터 로딩 중...")
    X, y, labels = load_dataset()

    num_classes = len(labels)
    print(f"\n총 {len(X)} 시퀀스, {num_classes}개 레이블")
    print(f"입력 형태: {X.shape}")

    counts = Counter(y.tolist())
    if min(counts.values()) < MIN_SEQUENCES_PER_LABEL:
        raise ValueError(
            "Each label needs at least 2 valid sequences for validation splitting."
        )

    val_count = max(num_classes, int(np.ceil(len(X) * VALIDATION_SPLIT)))
    val_count = min(val_count, len(X) - num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_count,
        stratify=y,
        random_state=42,
    )
    print(f"학습 {len(X_train)} / 검증 {len(X_val)}")

    model = build_model(num_classes)
    model.summary()

    print(f"\n학습 시작 (최대 {EPOCHS} 에포크)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=0,
    )
    print(
        f"학습 완료: {len(history.history['loss'])} epochs  |  "
        f"best val_accuracy={max(history.history['val_accuracy']):.1%}  |  "
        f"best val_loss={min(history.history['val_loss']):.4f}"
    )

    # Save labels alongside model
    import shutil
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"\n레이블 저장: {LABELS_PATH}")

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n최종 검증 정확도: {val_acc:.1%}  |  손실: {val_loss:.4f}")

    # Plot
    plot_path = os.path.join(MODEL_DIR, 'training_history.png')
    plot_history(history, plot_path)

    print(f"\n모델 저장: {MODEL_PATH}")
    print("학습 완료!")


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
