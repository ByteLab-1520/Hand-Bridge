r"""Presentation-machine preflight for Hand-Bridge on Windows.

Run after setup and before the event:
    .venv\Scripts\python.exe preflight_windows.py
"""

import json
import platform
import sys
import time

import cv2
import mediapipe
import numpy as np
import tensorflow as tf

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LABELS_PATH,
    MODEL_PATH,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
)
from model import make_inference_fn, run_inference
from utils import HolisticDetector, extract_landmarks, open_camera


def main() -> int:
    print("=== Hand-Bridge Windows preflight ===")
    print(f"Python {sys.version.split()[0]} | {platform.platform()}")
    print(
        f"TensorFlow {tf.__version__} | MediaPipe {mediapipe.__version__} "
        f"| OpenCV {cv2.__version__}"
    )

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)
    model = tf.keras.models.load_model(MODEL_PATH)
    expected_input = (None, SEQUENCE_LENGTH, NUM_FEATURES)
    if model.input_shape != expected_input or model.output_shape[-1] != len(labels):
        raise RuntimeError(
            f"모델/레이블 불일치: input={model.input_shape}, "
            f"output={model.output_shape}, labels={len(labels)}"
        )
    print(f"[PASS] 모델: {model.input_shape} -> {model.output_shape}, {len(labels)} labels")

    infer = make_inference_fn(model)
    sample = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
    for _ in range(3):
        run_inference(infer, sample)
    started = time.perf_counter()
    for _ in range(50):
        probs = run_inference(infer, sample)
    elapsed = time.perf_counter() - started
    inference_fps = 50 / elapsed
    if probs.shape != (len(labels),) or not np.isclose(probs.sum(), 1.0, atol=1e-4):
        raise RuntimeError("모델 확률 출력이 올바르지 않습니다.")
    print(f"[PASS] 최적화 추론: {inference_fps:.1f} calls/s")

    camera = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not camera.isOpened():
        print(f"[FAIL] 카메라 {CAMERA_INDEX}을(를) 열 수 없습니다.")
        return 1

    ok, frame = camera.read()
    if not ok or frame is None:
        camera.release()
        print("[FAIL] 카메라는 열렸지만 프레임을 읽지 못했습니다.")
        return 1
    actual = (frame.shape[1], frame.shape[0])
    print(f"[PASS] 카메라: {actual[0]}x{actual[1]} ({camera.getBackendName()})")

    detector = HolisticDetector()
    started = time.perf_counter()
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = extract_landmarks(results)
    detector_ms = (time.perf_counter() - started) * 1000
    detector.close()
    camera.release()
    if landmarks.shape != (NUM_FEATURES,):
        raise RuntimeError(f"랜드마크 형태 오류: {landmarks.shape}")
    print(f"[PASS] MediaPipe: {detector_ms:.1f} ms, landmarks={landmarks.shape}")
    print("\n모든 필수 검사를 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
