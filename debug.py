"""
Debug Mode — 실시간 웹캠 + Holistic 랜드마크 시각화 (MediaPipe Tasks API)

사용법:
    python debug.py

단축키:
    Q   — 종료
    F   — 얼굴 랜드마크 표시 토글
    P   — 포즈 랜드마크 표시 토글
    H   — 손 랜드마크 표시 토글
    S   — 현재 프레임 스크린샷 저장 (debug_capture.png)
"""

import os
import time

import cv2
import numpy as np

import config  # CUDA/Metal env vars
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    KEY_FACE_INDICES, NUM_KEY_FACE, POSE_INDICES, NUM_KEY_POSE,
)
from utils import put_korean_text, HolisticDetector

# ── 색상 ──────────────────────────────────────────────────────────────────────
C_GREEN  = (0, 255, 100)
C_ORANGE = (0, 160, 255)
C_RED    = (50,  50, 220)
C_WHITE  = (230, 230, 230)
C_DIM    = (100, 100, 100)
C_YELLOW = (0,   220, 220)
C_BLUE   = (255, 150,  50)

# ── 스켈레톤 연결 정의 ─────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# 상체 포즈 연결 (인덱스 0-16)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
]

# ── 랜드마크 그리기 헬퍼 ──────────────────────────────────────────────────────

def _draw_connections(frame, landmarks, connections, h, w, color, thickness=1):
    for a, b in connections:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        la, lb = landmarks[a], landmarks[b]
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(frame, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def _draw_dots(frame, landmarks, h, w, color, radius=3):
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_hands(frame, hand_result, h, w):
    if not hand_result or not hand_result.hand_landmarks:
        return
    for hand_lms in hand_result.hand_landmarks:
        _draw_connections(frame, hand_lms, HAND_CONNECTIONS, h, w, C_GREEN, 2)
        _draw_dots(frame, hand_lms, h, w, C_GREEN, 3)


def draw_pose(frame, pose_result, h, w):
    if not pose_result or not pose_result.pose_landmarks:
        return
    for pose_lms in pose_result.pose_landmarks:
        upper = [pose_lms[i] for i in POSE_INDICES if i < len(pose_lms)]
        _draw_connections(frame, pose_lms, POSE_CONNECTIONS, h, w, C_BLUE, 2)
        # 학습에 쓰이는 포인트 강조
        for idx in POSE_INDICES:
            if idx < len(pose_lms):
                lm = pose_lms[idx]
                if lm.visibility > 0.3:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, C_ORANGE, -1, cv2.LINE_AA)


def draw_face(frame, face_result, h, w):
    if not face_result or not face_result.face_landmarks:
        return
    for face_lms in face_result.face_landmarks:
        # 전체 윤곽 (얇게)
        for lm in face_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (60, 60, 60), -1)
        # 학습에 쓰이는 22개 포인트 강조 (노란색)
        for idx in KEY_FACE_INDICES:
            if idx < len(face_lms):
                lm = face_lms[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, C_YELLOW, -1, cv2.LINE_AA)


# ── 감지 상태 점 ──────────────────────────────────────────────────────────────

def _detection_dot(frame, x, y, detected: bool, label: str):
    color = C_GREEN if detected else C_RED
    cv2.circle(frame, (x, y), 6, color, -1)
    cv2.putText(frame, label, (x + 12, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    # HolisticDetector가 모델 다운로드 + 초기화를 모두 처리
    detector = HolisticDetector()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] 카메라를 열 수 없습니다 (CAMERA_INDEX={CAMERA_INDEX})")
        return

    show_face  = True
    show_pose  = True
    show_hands = True

    fps    = 0.0
    t_prev = time.time()
    print("=== Hand-Bridge Debug Mode ===")
    print("F: 얼굴  P: 포즈  H: 손  S: 스크린샷  Q: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        hand_result = results._hand
        face_result = results._face
        pose_result = results._pose

        # ── 랜드마크 그리기 ───────────────────────────────────────────────────
        from utils import LandmarkResults, draw_landmarks as _draw_all
        masked = LandmarkResults(
            hand_result if show_hands else None,
            face_result if show_face  else None,
            pose_result if show_pose  else None,
        )
        frame = _draw_all(frame, masked)
        # 학습 키포인트 강조
        if show_face and face_result and face_result.face_landmarks:
            for face_lms in face_result.face_landmarks:
                for idx in KEY_FACE_INDICES:
                    if idx < len(face_lms):
                        lm = face_lms[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, C_YELLOW, -1, cv2.LINE_AA)
        if show_pose and pose_result and pose_result.pose_landmarks:
            for pose_lms in pose_result.pose_landmarks:
                for idx in POSE_INDICES:
                    if idx < len(pose_lms) and pose_lms[idx].visibility > 0.3:
                        lm = pose_lms[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, C_ORANGE, -1, cv2.LINE_AA)

        # ── 감지 상태 패널 (우측 상단) ────────────────────────────────────────
        face_ok   = bool(face_result  and face_result.face_landmarks)
        pose_ok   = bool(pose_result  and pose_result.pose_landmarks)
        r_hand_ok = bool(hand_result  and any(
            hand_result.handedness[i][0].category_name == 'Right'
            for i in range(len(hand_result.hand_landmarks))
        )) if hand_result and hand_result.hand_landmarks else False
        l_hand_ok = bool(hand_result  and any(
            hand_result.handedness[i][0].category_name == 'Left'
            for i in range(len(hand_result.hand_landmarks))
        )) if hand_result and hand_result.hand_landmarks else False

        panel_x = w - 180
        cv2.rectangle(frame, (panel_x - 8, 8), (w - 8, 118), (20, 20, 20), -1)
        _detection_dot(frame, panel_x, 28,  face_ok,   "Face")
        _detection_dot(frame, panel_x, 52,  pose_ok,   "Pose")
        _detection_dot(frame, panel_x, 76,  r_hand_ok, "Right Hand")
        _detection_dot(frame, panel_x, 100, l_hand_ok, "Left Hand")

        # ── feature 충전율 ────────────────────────────────────────────────────
        active = (
            (63 if r_hand_ok else 0)
            + (63 if l_hand_ok else 0)
            + (NUM_KEY_FACE * 3 if face_ok else 0)
            + (NUM_KEY_POSE  * 3 if pose_ok else 0)
        )
        total = 63 + 63 + NUM_KEY_FACE * 3 + NUM_KEY_POSE * 3
        feat_color = C_GREEN if active == total else C_ORANGE
        cv2.putText(frame, f"features: {active}/{total}",
                    (panel_x - 8, 134),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, feat_color, 1, cv2.LINE_AA)

        # ── FPS ───────────────────────────────────────────────────────────────
        now    = time.time()
        fps    = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"FPS {fps:.0f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 2, cv2.LINE_AA)

        # ── 범례 + 토글 상태 (좌측 하단) ─────────────────────────────────────
        cv2.circle(frame, (10, h - 62), 4, C_YELLOW, -1)
        cv2.putText(frame, "= key face (22pt, 학습 사용)",
                    (20, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_YELLOW, 1)
        cv2.circle(frame, (10, h - 44), 4, C_ORANGE, -1)
        cv2.putText(frame, "= key pose (17pt, 학습 사용)",
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ORANGE, 1)

        frame = put_korean_text(
            frame,
            f"[F] 얼굴 {'ON' if show_face else 'OFF'}   "
            f"[P] 포즈 {'ON' if show_pose else 'OFF'}   "
            f"[H] 손 {'ON' if show_hands else 'OFF'}   "
            "[S] 저장   [Q] 종료",
            (10, h - 24), font_size=16, color=C_DIM, bg_color=None,
        )

        cv2.imshow("Hand-Bridge Debug", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f'):
            show_face  = not show_face
            print(f"얼굴: {'ON' if show_face else 'OFF'}")
        elif key == ord('p'):
            show_pose  = not show_pose
            print(f"포즈: {'ON' if show_pose else 'OFF'}")
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"손: {'ON' if show_hands else 'OFF'}")
        elif key == ord('s'):
            cv2.imwrite("debug_capture.png", frame)
            print("스크린샷 → debug_capture.png")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Debug 종료.")


if __name__ == '__main__':
    main()
