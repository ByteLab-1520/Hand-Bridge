"""
Data Collector — gather sign language training data.

Usage:
    python data_collector.py

Controls (while the OpenCV window is focused):
    SPACE   — start/pause/resume recording
    P       — pause/resume recording
    C       — cancel current sequence and discard collected frames
    Q       — quit
"""

import cv2
import numpy as np
import os
import json
import time

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    SEQUENCE_LENGTH, NUM_SEQUENCES,
    DATA_DIR,
)
from utils import extract_landmarks, draw_landmarks, put_korean_text, HolisticDetector, open_camera


def load_labels() -> dict[str, int]:
    labels_path = os.path.join(DATA_DIR, 'labels.json')
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_labels(labels: dict[str, int]) -> None:
    labels_path = os.path.join(DATA_DIR, 'labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def count_existing_sequences(label: str) -> int:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.endswith('.npy')])


def get_label_from_user(existing_labels: dict) -> str | None:
    """Simple terminal prompt — cv2 has no text input widget."""
    print("\n" + "="*50)
    if existing_labels:
        print("Existing labels:")
        for name, idx in sorted(existing_labels.items(), key=lambda x: x[1]):
            count = count_existing_sequences(name)
            print(f"  [{idx}] {name}  ({count} sequences)")
    print("\nEnter Korean label (e.g. 안녕하세요) — blank to quit: ", end='', flush=True)
    label = input().strip()
    return label if label else None


def collect_sequences(
    label: str,
    cap: cv2.VideoCapture,
    detector: HolisticDetector,
    num_sequences: int = NUM_SEQUENCES,
) -> int:
    """
    Interactively record `num_sequences` for `label`.
    Returns number of sequences actually saved.
    """
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    start_idx = count_existing_sequences(label)
    saved = 0

    while saved < num_sequences:
        seq_idx = start_idx + saved
        state = 'waiting'  # waiting | waiting_hand | countdown | recording | paused

        countdown_start = 0.0
        sequence: list[np.ndarray] = []
        frame_count = 0
        no_hand_count = 0  # Track consecutive frames without hand detection
        MIN_FRAMES = 20    # Minimum frames to collect before allowing early termination
        MAX_NO_HAND_FRAMES = 90  # ~3초 손 미인식 (30fps * 3초)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed.")
                return saved

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            frame = draw_landmarks(frame, results)
            has_hands = results.hand_detected

            now = time.time()

            if state == 'waiting':
                frame = put_korean_text(
                    frame,
                    f"수집 중: {label}  ({saved}/{num_sequences})",
                    (10, 20), font_size=28, color=(0, 255, 0),
                )
                frame = put_korean_text(
                    frame,
                    "SPACE: 시작  |  Q: 종료",
                    (10, 65), font_size=22, color=(255, 255, 0),
                )

            elif state == 'waiting_hand':
                if has_hands:
                    state = 'countdown'
                    countdown_start = time.time()
                else:
                    frame = put_korean_text(
                        frame,
                        "손을 카메라에 보여주세요",
                        (10, 20), font_size=28, color=(255, 165, 0),
                    )
                    frame = put_korean_text(
                        frame,
                        "손 인식 후 3초 뒤 녹화 시작",
                        (10, 65), font_size=22, color=(255, 255, 0),
                    )

            elif state == 'countdown':
                if not has_hands:
                    state = 'waiting_hand'
                else:
                    elapsed = now - countdown_start
                    remaining = 3 - int(elapsed)
                    if remaining <= 0:
                        state = 'recording'
                        sequence = []
                        frame_count = 0
                        no_hand_count = 0
                    else:
                        frame = put_korean_text(
                            frame,
                            str(remaining),
                            (w // 2 - 20, h // 2 - 40),
                            font_size=80, color=(0, 0, 255), bg_color=None,
                        )

            elif state == 'recording':
                if has_hands:
                    # 손이 감지됨 - 카운터 초기화
                    no_hand_count = 0
                else:
                    # 손이 감지되지 않음 - 카운터 증가
                    no_hand_count += 1
                    
                    # 90 프레임 이상 손이 없으면 조기 종료
                    if no_hand_count >= MAX_NO_HAND_FRAMES:
                        print(f"  [CANCEL] {label}/{seq_idx}.npy discarded (손 인식 없음 3초 초과)")
                        sequence = []
                        frame_count = 0
                        no_hand_count = 0
                        state = 'canceled'
                
                # 손이 감지된 경우에만 프레임 저장
                if has_hands:
                    landmarks = extract_landmarks(results)
                    sequence.append(landmarks)
                    frame_count += 1

                # Progress bar
                progress = int(frame_count / SEQUENCE_LENGTH * w)
                cv2.rectangle(frame, (0, h - 18), (progress, h), (0, 255, 0), -1)
                
                # 손 없음 상태 표시
                if no_hand_count > 0:
                    remaining_frames = MAX_NO_HAND_FRAMES - no_hand_count
                    frame = put_korean_text(
                        frame,
                        f"녹화 중... {frame_count}/{SEQUENCE_LENGTH}  (손 없음: {no_hand_count}/{MAX_NO_HAND_FRAMES})",
                        (10, 20), font_size=24, color=(255, 165, 0),
                    )
                else:
                    frame = put_korean_text(
                        frame,
                        f"녹화 중... {frame_count}/{SEQUENCE_LENGTH}",
                        (10, 20), font_size=24, color=(0, 0, 255),
                    )

                if frame_count >= SEQUENCE_LENGTH:
                    np.save(os.path.join(label_dir, f"{seq_idx}.npy"), np.array(sequence))
                    print(f"  [OK] {label}/{seq_idx}.npy saved (정상 완료: {frame_count} frames)")
                    saved += 1
                    state = 'saved'

            elif state == 'paused':
                frame = put_korean_text(
                    frame,
                    f"일시정지... {frame_count}/{SEQUENCE_LENGTH}",
                    (10, 20), font_size=28, color=(255, 165, 0),
                )
                frame = put_korean_text(
                    frame,
                    "SPACE/P: 재개  |  C: 취소  |  Q: 종료",
                    (10, 65), font_size=22, color=(255, 255, 0),
                )

            elif state == 'canceled':
                frame = put_korean_text(
                    frame, "시퀀스 취소됨", (10, 20), font_size=32, color=(255, 165, 0),
                )
                cv2.imshow('Sign Language Data Collector', frame)
                cv2.waitKey(600)
                break

            elif state == 'saved':
                frame = put_korean_text(
                    frame, "저장 완료!", (10, 20), font_size=32, color=(0, 255, 0),
                )
                cv2.imshow('Sign Language Data Collector', frame)
                cv2.waitKey(600)
                break  # inner while → next sequence

            cv2.imshow('Sign Language Data Collector', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and state == 'waiting':
                state = 'waiting_hand'
            elif key in (ord(' '), ord('p')) and state == 'recording':
                state = 'paused'
            elif key in (ord(' '), ord('p')) and state == 'paused':
                state = 'recording'
                no_hand_count = 0
            elif key == ord('c') and state in ('waiting_hand', 'countdown', 'recording', 'paused'):
                print(f"  [CANCEL] {label}/{seq_idx}.npy discarded")
                sequence = []
                frame_count = 0
                no_hand_count = 0
                state = 'canceled'
            elif key == ord('q'):
                return saved

    return saved


def main() -> None:
    detector = HolisticDetector()

    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        return

    print("=== 한국수어 데이터 수집기 ===")
    print(f"프레임 수: {SEQUENCE_LENGTH}  |  기본 수집량: {NUM_SEQUENCES} 시퀀스/레이블")

    labels = load_labels()

    while True:
        label = get_label_from_user(labels)
        if label is None:
            break

        # Ask how many sequences
        print(f"몇 개의 시퀀스를 수집할까요? (기본 {NUM_SEQUENCES}): ", end='', flush=True)
        n_input = input().strip()
        n_seqs = int(n_input) if n_input.isdigit() else NUM_SEQUENCES

        # Register label if new
        if label not in labels:
            labels[label] = len(labels)
            save_labels(labels)
            print(f"  새 레이블 등록: '{label}' → index {labels[label]}")

        print(f"\n[{label}] 수집 시작 — OpenCV 창에서 SPACE를 눌러 녹화하세요.")
        saved = collect_sequences(label, cap, detector, n_seqs)
        print(f"  총 {saved}개 시퀀스 저장 완료.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n수집 종료.")


if __name__ == '__main__':
    main()
