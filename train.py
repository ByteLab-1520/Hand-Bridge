"""
Train the sign language recognition model.

Usage:
    python train.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, MODEL_DIR, MODEL_PATH, LABELS_PATH,
    SEQUENCE_LENGTH, NUM_FEATURES,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT,
)
from model import build_model, get_callbacks


def load_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels_path = os.path.join(DATA_DIR, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"labels.json not found at {labels_path}. "
            "Run data_collector.py first."
        )

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels: dict[str, int] = json.load(f)

    X, y = [], []
    for label_name, label_idx in labels.items():
        label_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(label_dir):
            print(f"  [WARN] No data directory for '{label_name}', skipping.")
            continue

        seqs = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        if not seqs:
            print(f"  [WARN] No sequences for '{label_name}', skipping.")
            continue

        for fname in seqs:
            path = os.path.join(label_dir, fname)
            seq = np.load(path)
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                X.append(seq)
                y.append(label_idx)
            else:
                print(f"  [WARN] Bad shape {seq.shape} in {path}, skipping.")

        print(f"  Loaded {len(seqs)} sequences for '{label_name}' (idx={label_idx})")

    if not X:
        raise ValueError("No valid data found. Check your data directory.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels


def plot_history(history, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='train')
    axes[0].plot(history.history['val_accuracy'], label='val')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val')
    axes[1].set_title('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plot saved to {save_path}")


def main() -> None:
    print("=== 한국수어 모델 학습 ===\n")

    print("데이터 로딩 중...")
    X, y, labels = load_dataset()

    num_classes = len(labels)
    print(f"\n총 {len(X)} 시퀀스, {num_classes}개 레이블")
    print(f"입력 형태: {X.shape}")

    # Remap label indices to be contiguous 0..N-1
    unique_indices = sorted(set(y.tolist()))
    if unique_indices != list(range(num_classes)):
        print("  레이블 인덱스 재정렬 중...")
        remap = {old: new for new, old in enumerate(unique_indices)}
        y = np.array([remap[i] for i in y], dtype=np.int32)
        labels = {name: remap[idx] for name, idx in labels.items() if idx in remap}
        # Save updated labels
        with open(os.path.join(DATA_DIR, 'labels.json'), 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
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
        verbose=1,
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
    main()
