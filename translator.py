"""
Real-time Korean Sign Language Translator.

Usage:
    python translator.py

Controls:
    Q   — quit
    C   — clear current sentence
    S   — save sentence to output.txt
"""

import cv2
import numpy as np
import json
import os
import time
import collections
import tensorflow as tf

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    SEQUENCE_LENGTH, NUM_FEATURES,
    PREDICTION_THRESHOLD, STABLE_FRAMES,
    MIN_GESTURE_MOTION, MIN_GESTURE_DISPLACEMENT, INACTIVITY_CLEAR_SECONDS,
    MODEL_PATH, LABELS_PATH,
)
from utils import extract_landmarks, draw_landmarks, put_korean_text, HolisticDetector, open_camera
from model import make_inference_fn, run_inference


def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"모델 파일 없음: {MODEL_PATH}\n"
            "train.py를 먼저 실행하세요."
        )
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"레이블 파일 없음: {LABELS_PATH}\n"
            "train.py를 먼저 실행하세요."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        label_map: dict[str, int] = json.load(f)

    # index → Korean word
    idx_to_label = {v: k for k, v in label_map.items()}
    return model, idx_to_label


class Translator:
    def __init__(self, model, idx_to_label: dict[int, str]):
        self.model = model
        self.infer = make_inference_fn(model)
        self.idx_to_label = idx_to_label

        self.sequence: list[np.ndarray] = []
        self.sentence: list[str] = []
        self.last_prediction: str | None = None
        self.stable_count: int = 0
        self.last_added_word: str | None = None
        self.last_hand_time: float = time.time()  # 손 감지 타임스탬프
        self.last_input_time: float = time.time()

        # Smoothing: keep last few frame-level predictions
        self.pred_queue: collections.deque = collections.deque(maxlen=STABLE_FRAMES)

    def _gesture_has_motion(self) -> bool:
        hands = np.asarray(self.sequence, dtype=np.float32)[:, :126]
        motion = float(np.mean(np.abs(np.diff(hands, axis=0))))
        displacement = float(np.max(np.abs(hands[-1] - hands[0])))
        return (
            motion >= MIN_GESTURE_MOTION
            or displacement >= MIN_GESTURE_DISPLACEMENT
        )

    def process_frame(self, landmarks: np.ndarray, hand_detected: bool = True) -> tuple[str | None, float]:
        """
        Feed one frame of landmarks. Returns (predicted_label, confidence)
        when a stable prediction crosses the threshold, else (None, 0.0).
        
        Args:
            landmarks: 추출된 랜드마크 배열
            hand_detected: 손이 감지되었는지 여부
        """
        # 손이 감지되지 않으면
        if not hand_detected:
            self.sequence.clear()
            self.pred_queue.clear()
            self.clear_if_inactive()
            return None, 0.0
        
        self.last_hand_time = time.time()
        
        self.sequence.append(landmarks)
        if len(self.sequence) > SEQUENCE_LENGTH:
            self.sequence.pop(0)

        if len(self.sequence) < SEQUENCE_LENGTH:
            return None, 0.0

        if not self._gesture_has_motion():
            self.pred_queue.clear()
            self.clear_if_inactive()
            return None, 0.0

        self.last_input_time = time.time()

        probs = run_inference(self.infer, self.sequence)
        conf = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
        pred_label = self.idx_to_label.get(pred_idx, '?')

        self.pred_queue.append((pred_label, conf))

        # Require `STABLE_FRAMES` consecutive same prediction above threshold
        if len(self.pred_queue) < STABLE_FRAMES:
            return None, conf

        labels_in_queue = [p[0] for p in self.pred_queue]
        confs_in_queue = [p[1] for p in self.pred_queue]

        if len(set(labels_in_queue)) == 1 and min(confs_in_queue) >= PREDICTION_THRESHOLD:
            stable_label = labels_in_queue[0]
            avg_conf = float(np.mean(confs_in_queue))

            # Avoid repeating the same word back-to-back
            if stable_label != self.last_added_word:
                self.sentence.append(stable_label)
                self.last_added_word = stable_label
                self.pred_queue.clear()
                # 새로운 동작을 빠르게 인식하기 위해 시퀀스 초기화
                self.sequence.clear()
                return stable_label, avg_conf

        return None, conf

    def clear_sentence(self):
        self.sentence.clear()
        self.last_added_word = None
        self.pred_queue.clear()
        self.last_input_time = time.time()

    def clear_if_inactive(self):
        if (
            self.sentence
            and time.time() - self.last_input_time >= INACTIVITY_CLEAR_SECONDS
        ):
            self.clear_sentence()

    def get_sentence(self) -> str:
        return ' '.join(self.sentence)


def draw_ui(
    frame: np.ndarray,
    translator: Translator,
    current_conf: float,
    fps: float,
    idx_to_label: dict,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # ── Top bar: sentence ────────────────────────────────────────────────────
    sentence = translator.get_sentence() or '—'
    frame = put_korean_text(
        frame, f"번역: {sentence}",
        (10, 10), font_size=32, color=(255, 255, 255),
    )

    # ── Bottom bar: confidence + controls ────────────────────────────────────
    # Confidence bar background
    bar_y = h - 60
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 20), (60, 60, 60), -1)
    bar_width = int((w - 20) * min(current_conf, 1.0))
    bar_color = (0, 220, 0) if current_conf >= PREDICTION_THRESHOLD else (0, 140, 255)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 20), bar_color, -1)

    frame = put_korean_text(
        frame, f"신뢰도: {current_conf:.0%}  |  FPS: {fps:.0f}",
        (10, bar_y + 22), font_size=20, color=(200, 200, 200), bg_color=None,
    )
    frame = put_korean_text(
        frame, "Q: 종료  C: 문장 지우기  S: 저장",
        (10, h - 24), font_size=20, color=(200, 200, 200), bg_color=None,
    )

    # ── Sequence fill indicator (small dots) ─────────────────────────────────
    filled = len(translator.sequence)
    dot_y = h - 80
    for i in range(SEQUENCE_LENGTH):
        color = (0, 255, 0) if i < filled else (80, 80, 80)
        cv2.circle(frame, (10 + i * (w - 20) // SEQUENCE_LENGTH, dot_y), 3, color, -1)

    return frame


def save_sentence(sentence: str) -> None:
    path = os.path.join(os.path.dirname(__file__), 'output.txt')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(sentence + '\n')
    print(f"저장됨 → {path}: {sentence}")


def main() -> None:
    print("=== 한국수어 번역기 로딩 중... ===")
    model, idx_to_label = load_model_and_labels()
    print(f"모델 로드 완료. 인식 가능한 수어: {list(idx_to_label.values())}")

    translator = Translator(model, idx_to_label)

    hands = HolisticDetector()

    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. config.py의 CAMERA_INDEX를 확인하세요.")
        return

    print("번역기 시작. OpenCV 창에서 Q로 종료합니다.")

    fps_prev = time.time()
    fps = 0.0
    current_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)


        frame = draw_landmarks(frame, results)
        landmarks = extract_landmarks(results)

        _, current_conf = translator.process_frame(landmarks, hand_detected=results.hand_detected)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_prev, 1e-6))
        fps_prev = now

        frame = draw_ui(frame, translator, current_conf, fps, idx_to_label)

        cv2.imshow('한국수어 번역기', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            translator.clear_sentence()
            print("문장 초기화됨.")
        elif key == ord('s'):
            sentence = translator.get_sentence()
            if sentence and sentence != '—':
                save_sentence(sentence)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("번역기 종료.")


if __name__ == '__main__':
    main()
