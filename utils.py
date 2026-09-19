"""
Utility functions: Korean text rendering, landmark extraction, drawing.
Uses MediaPipe Tasks API (mediapipe >= 0.10.14).
"""

import os
import time
import urllib.request
import platform

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
from PIL import Image, ImageDraw, ImageFont

from config import (
    NUM_FEATURES, HAND_LANDMARKS, COORDS_PER_LANDMARK, MAX_HANDS,
    KEY_FACE_INDICES, NUM_KEY_FACE, POSE_INDICES, NUM_KEY_POSE,
    DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, MODELS_DIR,
    CAPTURE_TWO_HANDS, CAPTURE_WITH_FACE, CAPTURE_WITH_BODY, CAPTURE_ONE_HAND,
)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a webcam with a backend that is reliable in worker threads.

    Media Foundation can stall when opened from a background thread on some
    Windows systems.  DirectShow avoids that failure mode; MSMF remains the
    fallback for cameras that do not expose a DirectShow interface.
    """
    if platform.system() == "Windows":
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF)
    else:
        backends = (cv2.CAP_ANY,)

    camera = None
    for backend in backends:
        candidate = cv2.VideoCapture(index, backend)
        if candidate.isOpened():
            camera = candidate
            break
        candidate.release()

    if camera is None:
        camera = cv2.VideoCapture(index)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera

# ── Korean font ────────────────────────────────────────────────────────────────

def _find_korean_font() -> str | None:
    system = platform.system()
    candidates: list[str] = []
    if system == 'Darwin':
        candidates = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/NanumGothic.ttf',
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        ]
    elif system == 'Windows':
        win_fonts = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        candidates = [
            os.path.join(win_fonts, 'malgun.ttf'),
            os.path.join(win_fonts, 'gulim.ttc'),
        ]
    else:
        candidates = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


_KOREAN_FONT_PATH = _find_korean_font()


def put_korean_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 36,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] | None = (0, 0, 0),
    bg_alpha: float = 0.6,
) -> np.ndarray:
    if _KOREAN_FONT_PATH:
        font = ImageFont.truetype(_KOREAN_FONT_PATH, font_size)
    else:
        font = ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x, y = position

    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 6
        box_coords = (bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad)
        overlay = pil_img.copy()
        ImageDraw.Draw(overlay).rectangle(box_coords, fill=bg_color)
        pil_img = Image.blend(pil_img, overlay, bg_alpha)
        draw = ImageDraw.Draw(pil_img)

    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Model download ─────────────────────────────────────────────────────────────

_MODEL_URLS = {
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    ),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    ),
    'pose_landmarker_lite.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    ),
}


def ensure_model(filename: str) -> str:
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"모델 다운로드 중: {filename} ...")
        urllib.request.urlretrieve(_MODEL_URLS[filename], path)
        print(f"  → {path}")
    return path


# ── LandmarkResults container ──────────────────────────────────────────────────

class LandmarkResults:
    """
    Unified container wrapping Tasks API results for hand, face, and pose.

    Attributes mirror the Tasks API result objects:
        .hand_landmarks   list[list[NormalizedLandmark]]
        .handedness       list[list[Category]]
        .face_landmarks   list[list[NormalizedLandmark]]
        .pose_landmarks   list[list[NormalizedLandmark]]
        .hand_detected    bool
    """
    def __init__(self, hand_result, face_result, pose_result):
        self._hand = hand_result
        self._face = face_result
        self._pose = pose_result

    @property
    def hand_landmarks(self):
        return self._hand.hand_landmarks if self._hand else []

    @property
    def handedness(self):
        return self._hand.handedness if self._hand else []

    @property
    def face_landmarks(self):
        return self._face.face_landmarks if self._face else []

    @property
    def pose_landmarks(self):
        return self._pose.pose_landmarks if self._pose else []

    @property
    def hand_detected(self) -> bool:
        return bool(self._hand and self._hand.hand_landmarks)


# ── HolisticDetector ──────────────────────────────────────────────────────────

class HolisticDetector:
    """
    Wraps HandLandmarker + FaceLandmarker + PoseLandmarker from the Tasks API.

    Args:
        static: True  → IMAGE mode (non-sequential frames, e.g. video_importer)
                False → VIDEO mode (real-time webcam with tracking)
    """

    def __init__(self, static: bool = False, capture_mode: str = CAPTURE_TWO_HANDS):
        hand_model = ensure_model('hand_landmarker.task')

        mp_mode = RunningMode.IMAGE if static else RunningMode.VIDEO
        self._static       = static
        self._capture_mode = capture_mode
        self._t_start      = time.perf_counter()

        num_hands = 1 if capture_mode == CAPTURE_ONE_HAND else MAX_HANDS
        self._hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=hand_model),
                running_mode=mp_mode,
                num_hands=num_hands,
                min_hand_detection_confidence=DETECTION_CONFIDENCE,
                min_hand_presence_confidence=DETECTION_CONFIDENCE,
                min_tracking_confidence=TRACKING_CONFIDENCE,
            )
        )

        self._face = None
        if capture_mode == CAPTURE_WITH_FACE:
            face_model = ensure_model('face_landmarker.task')
            self._face = mp_vision.FaceLandmarker.create_from_options(
                mp_vision.FaceLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=face_model),
                    running_mode=mp_mode,
                    min_face_detection_confidence=DETECTION_CONFIDENCE,
                    min_face_presence_confidence=DETECTION_CONFIDENCE,
                    min_tracking_confidence=TRACKING_CONFIDENCE,
                )
            )

        self._pose = None
        if capture_mode == CAPTURE_WITH_BODY:
            pose_model = ensure_model('pose_landmarker_lite.task')
            self._pose = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=pose_model),
                    running_mode=mp_mode,
                    min_pose_detection_confidence=DETECTION_CONFIDENCE,
                    min_pose_presence_confidence=DETECTION_CONFIDENCE,
                    min_tracking_confidence=TRACKING_CONFIDENCE,
                )
            )

    def process(self, rgb_frame: np.ndarray) -> LandmarkResults:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._static:
            hand_result = self._hand.detect(mp_image)
            face_result = self._face.detect(mp_image) if self._face else None
            pose_result = self._pose.detect(mp_image) if self._pose else None
        else:
            ts = int((time.perf_counter() - self._t_start) * 1000)
            hand_result = self._hand.detect_for_video(mp_image, ts)
            face_result = self._face.detect_for_video(mp_image, ts) if self._face else None
            pose_result = self._pose.detect_for_video(mp_image, ts) if self._pose else None
        return LandmarkResults(hand_result, face_result, pose_result)

    def close(self):
        self._hand.close()
        if self._face:
            self._face.close()
        if self._pose:
            self._pose.close()


# ── Skeleton connections ───────────────────────────────────────────────────────

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
]


# ── Landmark extraction ────────────────────────────────────────────────────────

def extract_landmarks(results: LandmarkResults,
                      capture_mode: str = CAPTURE_TWO_HANDS) -> np.ndarray:
    """
    Extract a flat (243,) feature vector from LandmarkResults.

    Layout (always 243 elements; unused sections are zero-padded):
      [0   – 62 ]  right hand  (21 × 3)
      [63  – 125]  left hand   (21 × 3)  — zeroed in ONE_HAND mode
      [126 – 191]  key face    (22 × 3)  — only filled in WITH_FACE mode
      [192 – 242]  upper body pose (17 × 3) — only filled in WITH_BODY mode
    """
    landmarks = np.zeros(NUM_FEATURES, dtype=np.float32)

    # ── Hands ─────────────────────────────────────────────────────────────────
    if capture_mode == CAPTURE_ONE_HAND:
        # Use first detected hand, always place it in slot 0 (right-hand position)
        if results.hand_landmarks:
            hand_lms = results.hand_landmarks[0]
            for j, lm in enumerate(hand_lms):
                base = j * COORDS_PER_LANDMARK
                landmarks[base]     = lm.x
                landmarks[base + 1] = lm.y
                landmarks[base + 2] = lm.z
    else:
        for i, hand_lms in enumerate(results.hand_landmarks):
            if i >= len(results.handedness):
                break
            label = results.handedness[i][0].category_name  # 'Right' or 'Left'
            slot  = 0 if label == 'Right' else 1
            offset = slot * HAND_LANDMARKS * COORDS_PER_LANDMARK
            for j, lm in enumerate(hand_lms):
                base = offset + j * COORDS_PER_LANDMARK
                landmarks[base]     = lm.x
                landmarks[base + 1] = lm.y
                landmarks[base + 2] = lm.z

    # ── Key face landmarks ────────────────────────────────────────────────────
    face_offset = HAND_LANDMARKS * COORDS_PER_LANDMARK * MAX_HANDS  # 126
    if capture_mode == CAPTURE_WITH_FACE and results.face_landmarks:
        face_lms = results.face_landmarks[0]
        for slot, idx in enumerate(KEY_FACE_INDICES):
            if idx < len(face_lms):
                lm = face_lms[idx]
                base = face_offset + slot * COORDS_PER_LANDMARK
                landmarks[base]     = lm.x
                landmarks[base + 1] = lm.y
                landmarks[base + 2] = lm.z

    # ── Upper body pose ───────────────────────────────────────────────────────
    pose_offset = face_offset + NUM_KEY_FACE * COORDS_PER_LANDMARK  # 192
    if capture_mode == CAPTURE_WITH_BODY and results.pose_landmarks:
        pose_lms = results.pose_landmarks[0]
        for slot, idx in enumerate(POSE_INDICES):
            if idx < len(pose_lms):
                lm = pose_lms[idx]
                base = pose_offset + slot * COORDS_PER_LANDMARK
                landmarks[base]     = lm.x
                landmarks[base + 1] = lm.y
                landmarks[base + 2] = lm.z

    return landmarks


# ── Landmark drawing ───────────────────────────────────────────────────────────

def _draw_connections(frame, lms, connections, h, w, color, thickness=2):
    for a, b in connections:
        if a >= len(lms) or b >= len(lms):
            continue
        la, lb = lms[a], lms[b]
        # Skip if either endpoint is outside the visible frame
        if not (0.0 <= la.x <= 1.0 and 0.0 <= la.y <= 1.0
                and 0.0 <= lb.x <= 1.0 and 0.0 <= lb.y <= 1.0):
            continue
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(frame, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, results: LandmarkResults) -> np.ndarray:
    """Draw hand, face, and pose landmarks on frame."""
    h, w = frame.shape[:2]

    # Face — light dots for all landmarks
    for face_lms in results.face_landmarks:
        for lm in face_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (60, 100, 60), -1)

    # Pose — upper body skeleton
    for pose_lms in results.pose_landmarks:
        _draw_connections(frame, pose_lms, _POSE_CONNECTIONS, h, w, (200, 120, 50), 2)
        for idx in POSE_INDICES:
            if idx < len(pose_lms) and pose_lms[idx].visibility > 0.3:
                cx, cy = int(pose_lms[idx].x * w), int(pose_lms[idx].y * h)
                cv2.circle(frame, (cx, cy), 4, (200, 120, 50), -1, cv2.LINE_AA)

    # Hands — skeleton + dots
    for hand_lms in results.hand_landmarks:
        _draw_connections(frame, hand_lms, _HAND_CONNECTIONS, h, w, (0, 220, 100), 2)
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 120), -1, cv2.LINE_AA)

    return frame
