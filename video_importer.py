"""
MP4 영상 → 학습 데이터 변환기

videos/ 폴더 구조:
    videos/
    ├── 안녕하세요/
    │   ├── clip1.mp4
    │   ├── clip2.mp4
    │   └── ...
    ├── 감사합니다/
    │   └── video1.mp4
    └── ...

사용법:
    python video_importer.py              # 자동 변환 (손 감지율 50% 이상만 저장)
    python video_importer.py --review     # 영상마다 포즈 확인 후 수동 승인
    python video_importer.py --preview    # 저장 없이 결과만 확인
    python video_importer.py --videos_dir /다른/경로
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from config import (
    DATA_DIR,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
)
from utils import draw_landmarks, extract_landmarks, put_korean_text, HolisticDetector

VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
SUPPORTED_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

# 랜드마크 시각화에 쓸 스켈레톤 색상
_DETECTED_COLOR = (0, 255, 120)   # 초록 — 감지됨
_MISSING_COLOR  = (0, 80,  255)   # 주황 — 미감지


# ── 레이블 관리 ────────────────────────────────────────────────────────────────

def load_labels() -> dict[str, int]:
    path = os.path.join(DATA_DIR, 'labels.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_labels(labels: dict[str, int]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def count_existing(label: str) -> int:
    d = os.path.join(DATA_DIR, label)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.npy')])


# ── 포즈 추출 + 시각화 프레임 생성 ────────────────────────────────────────────

def _build_annotated_frames(
    video_path: str,
    detector: HolisticDetector,
) -> tuple[list[np.ndarray], list[np.ndarray], list[bool]]:
    """
    영상을 읽어 SEQUENCE_LENGTH 개 프레임을 균등 샘플링.

    반환:
        annotated_frames : 랜드마크가 그려진 BGR 프레임 리스트
        landmarks_list   : 각 프레임의 (126,) 특징 벡터 리스트
        detected_mask    : 프레임별 손 감지 여부 (bool)
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sample_indices = np.linspace(0, max(total - 1, 0), SEQUENCE_LENGTH, dtype=int)

    annotated_frames: list[np.ndarray] = []
    landmarks_list:   list[np.ndarray] = []
    detected_mask:    list[bool]       = []

    for fi, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            # 읽기 실패 → 빈 프레임
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            annotated_frames.append(blank)
            landmarks_list.append(np.zeros(NUM_FEATURES, dtype=np.float32))
            detected_mask.append(False)
            continue

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        lm      = extract_landmarks(results)

        detected = results.hand_detected

        # 랜드마크 그리기
        ann = frame.copy()
        ann = draw_landmarks(ann, results)

        # 프레임 번호 + 감지 상태 오버레이
        h, w = ann.shape[:2]
        status_color = _DETECTED_COLOR if detected else _MISSING_COLOR
        status_text  = "손 감지됨" if detected else "손 미감지"
        cv2.rectangle(ann, (0, h - 30), (w, h), (0, 0, 0), -1)
        ann = put_korean_text(
            ann,
            f"프레임 {fi+1}/{SEQUENCE_LENGTH}  |  {status_text}",
            (8, h - 28), font_size=18,
            color=status_color, bg_color=None,
        )

        # 감지율 progress bar (하단)
        detected_so_far = sum(detected_mask) + (1 if detected else 0)
        bar_w = int(detected_so_far / (fi + 1) * w)
        cv2.rectangle(ann, (0, h - 5), (bar_w, h), status_color, -1)

        annotated_frames.append(ann)
        landmarks_list.append(lm)
        detected_mask.append(detected)

    cap.release()
    return annotated_frames, landmarks_list, detected_mask


# ── 영상 1개 처리 ─────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    label: str,
    detector: HolisticDetector,
    review: bool,
    preview: bool,
    save_idx: int,
) -> str:
    """
    반환값: 'saved' | 'skipped' | 'rejected' | 'quit'
    """
    fname = os.path.basename(video_path)
    annotated, landmarks_list, detected_mask = _build_annotated_frames(
        video_path, detector
    )

    detection_rate = sum(detected_mask) / len(detected_mask)
    rate_str = f"{detection_rate:.0%}"

    # ── review 모드: OpenCV 창에서 재생 후 사용자 승인 ──────────────────────
    if review:
        decision = _review_video(annotated, detected_mask, label, fname, detection_rate)
        if decision == 'quit':
            return 'quit'
        if decision == 'reject':
            print(f"    [거부] {fname}  — 수동 거부")
            return 'rejected'
        # 'accept' → 아래로 이어서 저장
    else:
        # 자동 모드: 감지율 50% 미만이면 스킵
        if detection_rate < 0.5:
            print(f"    [스킵] {fname}  — 손 감지율 {rate_str} (50% 미만)")
            return 'skipped'

    # ── 저장 ──────────────────────────────────────────────────────────────────
    seq = np.array(landmarks_list, dtype=np.float32)  # (30, 126)

    if not preview:
        label_dir = os.path.join(DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        np.save(os.path.join(label_dir, f"{save_idx}.npy"), seq)

    suffix = "  (미리보기, 저장 안 함)" if preview else ""
    print(f"    [OK]   {fname}  — 손 감지율 {rate_str}"
          f"  ({sum(detected_mask)}/{SEQUENCE_LENGTH} 프레임){suffix}")
    return 'saved'


def _review_video(
    annotated_frames: list[np.ndarray],
    detected_mask: list[bool],
    label: str,
    fname: str,
    detection_rate: float,
) -> str:
    """
    OpenCV 창에서 포즈 애니메이션을 반복 재생하고 사용자 결정을 받는다.

    반환: 'accept' | 'reject' | 'quit'
    """
    WIN = 'Hand Pose Review'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 520)

    h, w = annotated_frames[0].shape[:2]
    frame_idx = 0
    paused = False

    while True:
        frame = annotated_frames[frame_idx].copy()

        # 상단 정보 패널
        panel = np.zeros((64, w, 3), dtype=np.uint8)
        frame = np.vstack([panel, frame])

        frame = put_korean_text(
            frame,
            f"레이블: {label}  |  파일: {fname}",
            (8, 4), font_size=20, color=(255, 255, 255), bg_color=None,
        )
        rate_color = _DETECTED_COLOR if detection_rate >= 0.5 else _MISSING_COLOR
        frame = put_korean_text(
            frame,
            f"손 감지율: {detection_rate:.0%}  ({sum(detected_mask)}/{len(detected_mask)}프레임)",
            (8, 30), font_size=18, color=rate_color, bg_color=None,
        )
        frame = put_korean_text(
            frame,
            "Y: 저장   N: 거부   SPACE: 일시정지   Q: 전체종료",
            (8, 52), font_size=16, color=(180, 180, 180), bg_color=None,
        )

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(60) & 0xFF   # ~16 fps 재생

        if key == ord('y'):
            cv2.destroyWindow(WIN)
            return 'accept'
        elif key == ord('n'):
            cv2.destroyWindow(WIN)
            return 'reject'
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return 'quit'
        elif key == ord(' '):
            paused = not paused

        if not paused:
            frame_idx = (frame_idx + 1) % len(annotated_frames)


# ── 레이블 폴더 처리 ──────────────────────────────────────────────────────────

def process_label(
    label: str,
    video_files: list[str],
    labels: dict[str, int],
    detector: HolisticDetector,
    review: bool,
    preview: bool,
) -> tuple[int, int, bool]:
    """
    반환: (성공 수, 실패 수, 계속할지 여부)
    """
    if label not in labels:
        labels[label] = len(labels)

    start_idx = count_existing(label)
    success = 0
    fail = 0

    for vpath in video_files:
        result = process_video(
            vpath, label, detector,
            review=review, preview=preview,
            save_idx=start_idx + success,
        )
        if result == 'quit':
            return success, fail, False
        elif result == 'saved':
            success += 1
        else:
            fail += 1

    return success, fail, True


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='MP4 → 학습 데이터 변환기')
    parser.add_argument('--videos_dir', default=VIDEOS_DIR,
                        help='영상 폴더 경로 (기본: ./videos)')
    parser.add_argument('--review', action='store_true',
                        help='영상마다 손 포즈를 확인하고 Y/N으로 수동 승인')
    parser.add_argument('--preview', action='store_true',
                        help='변환만 확인하고 저장하지 않음')
    args = parser.parse_args()

    videos_root = args.videos_dir
    if not os.path.isdir(videos_root):
        print(f"[오류] 폴더가 없습니다: {videos_root}")
        print("videos/ 폴더 구조 예시:\n")
        print("  videos/")
        print("  ├── 안녕하세요/")
        print("  │   ├── clip1.mp4")
        print("  │   └── clip2.mp4")
        print("  └── 감사합니다/")
        print("      └── video1.mp4")
        sys.exit(1)

    label_dirs = sorted([
        d for d in os.listdir(videos_root)
        if os.path.isdir(os.path.join(videos_root, d)) and not d.startswith('.')
    ])

    if not label_dirs:
        print(f"[오류] {videos_root} 안에 레이블 폴더가 없습니다.")
        sys.exit(1)

    print("=== MP4 → 학습 데이터 변환 ===")
    if args.review:
        print("※ 리뷰 모드 — 영상마다 손 포즈를 확인하고 Y/N으로 승인합니다\n")
    if args.preview:
        print("※ 미리보기 모드 — 실제로 저장하지 않습니다\n")

    labels = load_labels()

    detector = HolisticDetector(static=True)  # IMAGE mode: frames are non-sequential

    total_ok = 0
    total_fail = 0

    for label in label_dirs:
        label_path = os.path.join(videos_root, label)
        video_files = sorted([
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ])

        if not video_files:
            print(f"\n[{label}] — 지원 영상 없음 (건너뜀)")
            continue

        existing = count_existing(label)
        print(f"\n[{label}]  영상 {len(video_files)}개  |  기존 시퀀스 {existing}개")

        ok, fail, keep_going = process_label(
            label, video_files, labels, detector,
            review=args.review, preview=args.preview,
        )
        total_ok   += ok
        total_fail += fail

        if not keep_going:
            print("\n[사용자 종료]")
            break

    detector.close()

    if not args.preview:
        save_labels(labels)
        print(f"\n레이블 저장 완료: {os.path.join(DATA_DIR, 'labels.json')}")

    print(f"\n=== 완료 ===")
    print(f"저장: {total_ok}개  |  스킵/거부: {total_fail}개")

    if total_ok > 0 and not args.preview:
        print("\n다음 단계: python train.py")


if __name__ == '__main__':
    main()
