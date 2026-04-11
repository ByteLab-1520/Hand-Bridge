"""
한국수어 번역기 — Pixel UI
실행: python gui.py
"""

import collections
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time

import cv2
import customtkinter as ctk
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import tkinter as tk

import config  # CUDA/Metal 비활성화 env vars
from config import (
    CAMERA_INDEX, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SEQUENCE_LENGTH, NUM_FEATURES, STABLE_FRAMES, PREDICTION_THRESHOLD,
    MODEL_PATH, LABELS_PATH, DATA_DIR, NUM_SEQUENCES,
    CAPTURE_MODES, DEFAULT_CAPTURE_MODE,
    CAPTURE_TWO_HANDS, CAPTURE_WITH_FACE, CAPTURE_WITH_BODY, CAPTURE_ONE_HAND,
)
from utils import extract_landmarks, draw_landmarks, HolisticDetector

# ── 60-30-10 색상 규칙 ────────────────────────────────────────────────────────
# 60%  지배색 (배경, 대형 표면)   → 딥 다크 네이비
# 30%  보조색 (패널, 카드, 버튼)  → 다크 슬레이트 블루
# 10%  강조색 (하이라이트, CTA)   → 네온 그린
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# 60% ── 지배색
C_BG        = "#0b0d14"   # 메인 배경
C_BG_ALT    = "#0f1219"   # 약간 밝은 배경 (구분용)

# 30% ── 보조색
C_PANEL     = "#1c2333"   # 패널 배경
C_SURFACE   = "#243044"   # 카드 / 입력 배경
C_BORDER    = "#2e3f5c"   # 테두리
C_BTN       = "#1c2333"   # 버튼 기본
C_BTN_HOVER = "#2a3a55"   # 버튼 호버
C_TEXT      = "#dde6f5"   # 기본 텍스트 (30% 위)
C_TEXT_DIM  = "#4a6080"   # 보조 텍스트

# 10% ── 강조색
C_ACCENT    = "#00ff88"   # 네온 그린 메인
C_ACCENT2   = "#00cc6a"   # 강조 호버 / 보조
C_BTN_ACT   = "#00ff88"

# 기능색 (팔레트 외 — 경고/오류 전용)
C_WARN      = "#ffaa00"
C_ERROR     = "#ff4455"

CAM_W, CAM_H = 560, 315   # 16:9 프리뷰
SIDEBAR_W    = 170
WIN_W, WIN_H = 1040, 680

# ── 픽셀 폰트 ─────────────────────────────────────────────────────────────────
_MONO = {
    "Darwin":  "Courier",
    "Windows": "Courier New",
    "Linux":   "Courier New",
}.get(platform.system(), "Courier New")

_KR = {
    "Darwin":  "Apple SD Gothic Neo",
    "Windows": "Malgun Gothic",
    "Linux":   "Noto Sans CJK KR",
}.get(platform.system(), "")


def pf(size=12, bold=False):
    """픽셀(모노) 폰트"""
    return ctk.CTkFont(family=_MONO, size=size, weight="bold" if bold else "normal")


def kf(size=13, bold=False):
    """한글 폰트"""
    return ctk.CTkFont(family=_KR, size=size, weight="bold" if bold else "normal")


def _px_btn(parent, text, command=None, width=140, accent=False, warn=False):
    """픽셀 스타일 버튼"""
    fg    = C_ACCENT if accent else (C_WARN if warn else C_BTN)
    hover = "#00cc66" if accent else ("#cc8800" if warn else C_BTN_HOVER)
    tc    = C_BG if accent or warn else C_TEXT
    return ctk.CTkButton(
        parent, text=text, command=command, width=width,
        font=kf(12, bold=True),
        fg_color=fg, hover_color=hover, text_color=tc,
        corner_radius=0, border_width=1, border_color=C_BORDER,
    )


def _px_label(parent, text, size=12, color=None, mono=False):
    font = pf(size) if mono else kf(size)
    return ctk.CTkLabel(parent, text=text, font=font,
                        text_color=color or C_TEXT)


def _px_frame(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=C_SURFACE, corner_radius=0,
                        border_width=1, border_color=C_BORDER, **kw)


def _px_textbox(parent, height=200, font=None):
    return ctk.CTkTextbox(
        parent, height=height,
        font=font or pf(11),
        fg_color=C_PANEL, text_color=C_ACCENT,
        corner_radius=0, border_width=1, border_color=C_BORDER,
    )


def _px_progress(parent, width=300):
    return ctk.CTkProgressBar(
        parent, width=width, height=12,
        fg_color=C_BORDER, progress_color=C_ACCENT,
        corner_radius=0,
    )


# ── 카메라 워커 ───────────────────────────────────────────────────────────────
class CameraWorker(threading.Thread):
    """백그라운드에서 카메라 캡처 + MediaPipe 실행."""

    def __init__(self, out_q: queue.Queue, stop_evt: threading.Event,
                 capture_mode: str = DEFAULT_CAPTURE_MODE):
        super().__init__(daemon=True)
        self.out_q        = out_q
        self.stop_evt     = stop_evt
        self.capture_mode = capture_mode

    def run(self):
        detector = HolisticDetector(capture_mode=self.capture_mode)
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while not self.stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            lm      = extract_landmarks(results, capture_mode=self.capture_mode)
            ann     = draw_landmarks(frame.copy(), results)
            # Center crosshair
            fh, fw = ann.shape[:2]
            cx, cy, arm = fw // 2, fh // 2, 24
            _cross_color = (0, 255, 136)  # neon green (BGR)
            cv2.line(ann, (cx - arm, cy), (cx + arm, cy), _cross_color, 1, cv2.LINE_AA)
            cv2.line(ann, (cx, cy - arm), (cx, cy + arm), _cross_color, 1, cv2.LINE_AA)
            disp    = cv2.cvtColor(cv2.resize(ann, (CAM_W, CAM_H)), cv2.COLOR_BGR2RGB)
            try:
                self.out_q.put_nowait((disp, lm, results.hand_detected))
            except queue.Full:
                pass

        cap.release()
        detector.close()


# ── 카메라 패널 (공유 위젯) ───────────────────────────────────────────────────
class CameraPanel(ctk.CTkFrame):
    """카메라 프리뷰 + 픽셀 오버레이를 표시하는 위젯."""

    def __init__(self, parent):
        super().__init__(parent, fg_color=C_PANEL, corner_radius=0,
                         border_width=1, border_color=C_BORDER,
                         width=CAM_W, height=CAM_H + 28)
        self.pack_propagate(False)

        # 픽셀 헤더 바
        hdr = ctk.CTkFrame(self, fg_color=C_SURFACE, corner_radius=0, height=24)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="[ CAMERA FEED ]", font=pf(10),
                     text_color=C_ACCENT).pack(side="left", padx=6)
        self._fps_lbl = ctk.CTkLabel(hdr, text="00 FPS", font=pf(10),
                                     text_color=C_TEXT_DIM)
        self._fps_lbl.pack(side="right", padx=6)

        # 영상 레이블
        self._lbl = ctk.CTkLabel(self, text="■ NO SIGNAL ■", font=pf(14, bold=True),
                                 text_color=C_TEXT_DIM, fg_color=C_BG,
                                 width=CAM_W, height=CAM_H)
        self._lbl.pack()

        # 오버레이 (카운트다운 등) — place() 사용
        self._overlay = ctk.CTkLabel(self._lbl, text="", font=pf(72, bold=True),
                                     text_color=C_WARN, fg_color="transparent")
        self._overlay.place(relx=0.5, rely=0.5, anchor="center")

        self._prev_time = time.time()
        self._fps = 0.0

    def update_frame(self, frame_rgb: np.ndarray):
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._prev_time, 1e-6)
        self._prev_time = now
        self._fps_lbl.configure(text=f"{self._fps:02.0f} FPS")

        img = ctk.CTkImage(Image.fromarray(frame_rgb), size=(CAM_W, CAM_H))
        self._lbl.configure(image=img, text="")
        self._lbl._image = img

    def set_overlay(self, text: str, color: str = C_WARN):
        self._overlay.configure(text=text, text_color=color)


# ═══════════════════════════════════════════════════════════════════════════════
# 번역기 페이지
# ═══════════════════════════════════════════════════════════════════════════════
class TranslatorPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C_BG, corner_radius=0)
        self.app = app
        self._q: queue.Queue   = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._model = None
        self._idx2lbl: dict[int, str] = {}
        self._seq:  list[np.ndarray]  = []
        self._buf:  collections.deque = collections.deque(maxlen=STABLE_FRAMES)
        self._sentence: list[str]     = []
        self._last_word: str | None   = None
        self._last_hand_time: float   = 0.0
        self._build()
        self._load_model()

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    def _build(self):
        row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        row.pack(fill="both", expand=True, padx=12, pady=12)

        # 왼쪽: 카메라
        left = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        left.pack(side="left", fill="y")
        self._cam = CameraPanel(left)
        self._cam.pack()

        # 오른쪽: 정보
        right = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        # 현재 인식 단어
        _px_label(right, "[ CURRENT SIGN ]", 10, C_ACCENT, mono=True).pack(anchor="w")
        self._cur_lbl = _px_label(right, "—", 36, C_ACCENT)
        self._cur_lbl.pack(anchor="w", pady=(2, 10))

        # 신뢰도
        _px_label(right, "[ CONFIDENCE ]", 10, C_ACCENT, mono=True).pack(anchor="w")
        self._conf_bar = _px_progress(right, width=320)
        self._conf_bar.set(0)
        self._conf_bar.pack(anchor="w", pady=(2, 2))
        self._conf_txt = _px_label(right, "0%", 11, C_TEXT_DIM, mono=True)
        self._conf_txt.pack(anchor="w", pady=(0, 10))

        # 번역 결과
        _px_label(right, "[ TRANSLATION ]", 10, C_ACCENT, mono=True).pack(anchor="w")
        self._sent_box = _px_textbox(right, height=100, font=kf(18, bold=True))
        self._sent_box.pack(fill="x", pady=(2, 12))

        # 캡처 모드
        _px_label(right, "[ CAPTURE MODE ]", 10, C_ACCENT, mono=True).pack(anchor="w", pady=(0, 4))
        self._mode_seg = ctk.CTkSegmentedButton(
            right, values=CAPTURE_MODES,
            command=self._on_mode_change,
            font=kf(12),
            fg_color=C_PANEL, selected_color=C_ACCENT2, selected_hover_color=C_ACCENT,
            unselected_color=C_BTN, unselected_hover_color=C_BTN_HOVER,
            text_color=C_TEXT, text_color_disabled=C_TEXT_DIM,
            corner_radius=0, border_width=1,
        )
        self._mode_seg.set(DEFAULT_CAPTURE_MODE)
        self._mode_seg.pack(anchor="w", pady=(0, 12))

        # 모델 상태
        self._status = _px_label(right, "Loading...", 11, C_TEXT_DIM, mono=True)
        self._status.pack(anchor="w", pady=(0, 12))

        # 버튼
        btn_row = ctk.CTkFrame(right, fg_color="transparent", corner_radius=0)
        btn_row.pack(anchor="w")
        _px_btn(btn_row, "[ CLR ] 초기화", self._clear, 140).pack(side="left", padx=(0, 6))
        _px_btn(btn_row, "[ SAV ] 저장",   self._save,  130, accent=True).pack(side="left")

    def _load_model(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            self._status.configure(text=">> 모델 없음 — [학습] 탭에서 먼저 학습하세요",
                                   text_color=C_WARN)
            return
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                lmap = json.load(f)
            self._idx2lbl = {v: k for k, v in lmap.items()}
            self._status.configure(
                text=f">> MODEL OK  |  {len(self._idx2lbl)} labels", text_color=C_ACCENT)
        except Exception as e:
            self._status.configure(text=f">> ERROR: {e}", text_color=C_ERROR)

    # ── 페이지 진입/이탈 ──────────────────────────────────────────────────────
    def on_activate(self):
        self._stop.clear()
        CameraWorker(self._q, self._stop,
                     capture_mode=self._mode_seg.get()).start()
        self._update()

    def on_deactivate(self):
        self._stop.set()

    def _on_mode_change(self, value: str):
        self._stop.set()
        self.after(200, self._restart_worker)

    def _restart_worker(self):
        self._stop.clear()
        CameraWorker(self._q, self._stop,
                     capture_mode=self._mode_seg.get()).start()
        self._update()

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    def _update(self):
        if self._stop.is_set():
            return
        try:
            frame_rgb, lm, hand_detected = self._q.get_nowait()
            self._cam.update_frame(frame_rgb)
            if self._model is not None:
                self._infer(lm, hand_detected)
        except queue.Empty:
            pass
        self.after(30, self._update)

    def _infer(self, lm: np.ndarray, hand_detected: bool):
        if not hand_detected:
            self._seq.clear()
            # 3초간 손 없으면 자동 CLR
            if self._sentence and time.time() - self._last_hand_time > 3.0:
                self._clear()
            return

        self._last_hand_time = time.time()
        self._seq.append(lm)
        if len(self._seq) > SEQUENCE_LENGTH:
            self._seq.pop(0)
        if len(self._seq) < SEQUENCE_LENGTH:
            return

        probs = self._model.predict(np.expand_dims(self._seq, 0), verbose=0)[0]
        conf  = float(np.max(probs))
        word  = self._idx2lbl.get(int(np.argmax(probs)), '?')

        self._conf_bar.set(min(conf, 1.0))
        self._conf_txt.configure(text=f"{conf:.0%}")
        self._buf.append((word, conf))

        if len(self._buf) == STABLE_FRAMES:
            words = [p[0] for p in self._buf]
            confs = [p[1] for p in self._buf]
            if len(set(words)) == 1 and min(confs) >= PREDICTION_THRESHOLD:
                w = words[0]
                self._sentence.append(w)
                self._last_word = w
                # 다음 동작 즉시 인식을 위해 버퍼·시퀀스 초기화
                self._buf.clear()
                self._seq.clear()
                self._cur_lbl.configure(text=w)
                self._refresh_box()

    def _refresh_box(self):
        text = ' '.join(self._sentence) or '—'
        self._sent_box.configure(state="normal")
        self._sent_box.delete("1.0", "end")
        self._sent_box.insert("1.0", text)
        self._sent_box.configure(state="disabled")

    def _clear(self):
        self._sentence.clear()
        self._last_word = None
        self._buf.clear()
        self._cur_lbl.configure(text="—")
        self._refresh_box()

    def _save(self):
        text = ' '.join(self._sentence)
        if not text:
            return
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output.txt')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        messagebox.showinfo("저장 완료", f"output.txt 저장:\n{text}")


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 수집 페이지
# ═══════════════════════════════════════════════════════════════════════════════
class CollectorPage(ctk.CTkFrame):
    _IDLE       = 'IDLE'
    _COUNTDOWN  = 'COUNTDOWN'
    _RECORDING  = 'RECORDING'

    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C_BG, corner_radius=0)
        self.app = app
        self._q:    queue.Queue   = queue.Queue(maxsize=2)
        self._stop  = threading.Event()
        self._state = self._IDLE
        self._cd_start   = 0.0
        self._sequence:  list[np.ndarray] = []
        self._saved_count = 0
        self._labels: dict[str, int] = {}
        self._build()

    def _build(self):
        row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        row.pack(fill="both", expand=True, padx=12, pady=12)

        # 왼쪽: 카메라
        left = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        left.pack(side="left", fill="y")
        self._cam = CameraPanel(left)
        self._cam.pack()

        # 오른쪽: 컨트롤
        right = ctk.CTkFrame(row, fg_color="transparent", corner_radius=0)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        # 캡처 모드
        _px_label(right, "[ CAPTURE MODE ]", 10, C_ACCENT, mono=True).pack(anchor="w")
        self._mode_seg = ctk.CTkSegmentedButton(
            right, values=CAPTURE_MODES,
            command=self._on_mode_change,
            font=kf(12),
            fg_color=C_PANEL, selected_color=C_ACCENT2, selected_hover_color=C_ACCENT,
            unselected_color=C_BTN, unselected_hover_color=C_BTN_HOVER,
            text_color=C_TEXT, text_color_disabled=C_TEXT_DIM,
            corner_radius=0, border_width=1,
        )
        self._mode_seg.set(DEFAULT_CAPTURE_MODE)
        self._mode_seg.pack(anchor="w", pady=(2, 12))

        # 레이블 입력
        _px_label(right, "[ LABEL ]", 10, C_ACCENT, mono=True).pack(anchor="w")
        self._label_entry = ctk.CTkEntry(
            right, placeholder_text="한국어 수어 단어 입력",
            font=kf(14), fg_color=C_SURFACE, border_color=C_BORDER,
            corner_radius=0, border_width=1, text_color=C_TEXT, width=280,
        )
        self._label_entry.pack(anchor="w", pady=(2, 10))

        # 수집 개수
        _px_label(right, "[ SEQUENCES ]", 10, C_ACCENT, mono=True).pack(anchor="w")
        self._n_entry = ctk.CTkEntry(
            right, placeholder_text=str(NUM_SEQUENCES),
            font=pf(13), fg_color=C_SURFACE, border_color=C_BORDER,
            corner_radius=0, border_width=1, text_color=C_TEXT, width=100,
        )
        self._n_entry.pack(anchor="w", pady=(2, 12))

        # 진행 상황
        _px_label(right, "[ PROGRESS ]", 10, C_ACCENT, mono=True).pack(anchor="w")
        self._prog_bar = _px_progress(right, width=280)
        self._prog_bar.set(0)
        self._prog_bar.pack(anchor="w", pady=(2, 2))
        self._prog_txt = _px_label(right, "0 / 0", 11, C_TEXT_DIM, mono=True)
        self._prog_txt.pack(anchor="w", pady=(0, 12))

        # 상태
        self._state_lbl = _px_label(right, ">> IDLE", 12, C_TEXT_DIM, mono=True)
        self._state_lbl.pack(anchor="w", pady=(0, 12))

        # 버튼
        self._rec_btn = _px_btn(right, "[ REC ] 녹화 시작", self._on_record, 180, accent=True)
        self._rec_btn.pack(anchor="w")

    # ── 레이블 파일 로드/저장 ─────────────────────────────────────────────────
    def _load_labels(self):
        path = os.path.join(DATA_DIR, 'labels.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self._labels = json.load(f)

    def _save_labels(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, 'labels.json'), 'w', encoding='utf-8') as f:
            json.dump(self._labels, f, ensure_ascii=False, indent=2)

    # ── 페이지 진입/이탈 ──────────────────────────────────────────────────────
    def on_activate(self):
        self._load_labels()
        self._stop.clear()
        self._state = self._IDLE
        self._saved_count = 0
        CameraWorker(self._q, self._stop,
                     capture_mode=self._mode_seg.get()).start()
        self._update()

    def on_deactivate(self):
        self._stop.set()

    def _on_mode_change(self, value: str):
        if self._state != self._IDLE:
            return  # don't change mode while recording
        self._stop.set()
        self.after(200, self._restart_worker)

    def _restart_worker(self):
        self._stop.clear()
        CameraWorker(self._q, self._stop,
                     capture_mode=self._mode_seg.get()).start()
        self._update()

    # ── 녹화 버튼 ─────────────────────────────────────────────────────────────
    def _on_record(self):
        if self._state != self._IDLE:
            return
        label = self._label_entry.get().strip()
        if not label:
            messagebox.showwarning("입력 오류", "레이블을 입력하세요.")
            return
        n_str = self._n_entry.get().strip()
        self._target = int(n_str) if n_str.isdigit() else NUM_SEQUENCES
        self._current_label = label
        self._saved_count = 0
        self._prog_bar.set(0)
        self._prog_txt.configure(text=f"0 / {self._target}")

        # 새 레이블 등록
        if label not in self._labels:
            self._labels[label] = len(self._labels)
            self._save_labels()

        self._start_next_sequence()

    def _start_next_sequence(self):
        if self._saved_count >= self._target:
            self._state = self._IDLE
            self._state_lbl.configure(text=f">> DONE  {self._saved_count} sequences saved",
                                      text_color=C_ACCENT)
            self._cam.set_overlay("")
            self._rec_btn.configure(state="normal")
            return
        self._state = self._COUNTDOWN
        self._cd_start = time.time()
        self._rec_btn.configure(state="disabled")

    def _save_sequence(self):
        label_dir = os.path.join(DATA_DIR, self._current_label)
        os.makedirs(label_dir, exist_ok=True)
        existing = len([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        np.save(os.path.join(label_dir, f"{existing}.npy"),
                np.array(self._sequence, dtype=np.float32))
        self._saved_count += 1
        self._prog_bar.set(self._saved_count / self._target)
        self._prog_txt.configure(text=f"{self._saved_count} / {self._target}")

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    def _update(self):
        if self._stop.is_set():
            return

        # 카운트다운 (프레임 없어도 진행)
        if self._state == self._COUNTDOWN:
            elapsed = time.time() - self._cd_start
            rem = 3 - int(elapsed)
            if rem <= 0:
                self._state = self._RECORDING
                self._sequence = []
                self._cam.set_overlay("●", C_ERROR)
                self._state_lbl.configure(text=">> RECORDING...", text_color=C_ERROR)
            else:
                self._cam.set_overlay(str(rem), C_WARN)
                self._state_lbl.configure(text=f">> COUNTDOWN {rem}",
                                          text_color=C_WARN)

        try:
            frame_rgb, lm, hand_detected = self._q.get_nowait()
            self._cam.update_frame(frame_rgb)

            if self._state == self._RECORDING:
                self._sequence.append(lm)
                pct = len(self._sequence) / SEQUENCE_LENGTH
                self._prog_bar.set(self._saved_count / self._target +
                                   pct / self._target)
                if len(self._sequence) >= SEQUENCE_LENGTH:
                    self._save_sequence()
                    self._cam.set_overlay("OK", C_ACCENT)
                    self._state = self._IDLE
                    self.after(500, self._start_next_sequence)

        except queue.Empty:
            pass

        self.after(30, self._update)


# ═══════════════════════════════════════════════════════════════════════════════
# 영상 가져오기 페이지
# ═══════════════════════════════════════════════════════════════════════════════
class ImporterPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C_BG, corner_radius=0)
        self.app = app
        self._log_q: queue.Queue = queue.Queue()
        self._running = False
        self._build()

    def _build(self):
        pad = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        pad.pack(fill="both", expand=True, padx=16, pady=12)

        # 제목
        _px_label(pad, "[ VIDEO IMPORTER ]", 13, C_ACCENT, mono=True).pack(anchor="w", pady=(0, 10))

        # 폴더 선택
        row1 = ctk.CTkFrame(pad, fg_color="transparent", corner_radius=0)
        row1.pack(fill="x", pady=(0, 8))
        _px_label(row1, "VIDEOS DIR :", 11, C_TEXT_DIM, mono=True).pack(side="left", padx=(0, 6))
        self._dir_var = tk.StringVar(value=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'videos'))
        self._dir_entry = ctk.CTkEntry(
            row1, textvariable=self._dir_var, width=380,
            font=pf(11), fg_color=C_SURFACE, border_color=C_BORDER,
            corner_radius=0, border_width=1, text_color=C_TEXT,
        )
        self._dir_entry.pack(side="left", padx=(0, 6))
        _px_btn(row1, "[ ... ]", self._browse, 70).pack(side="left")

        # 옵션
        row2 = ctk.CTkFrame(pad, fg_color="transparent", corner_radius=0)
        row2.pack(fill="x", pady=(0, 12))
        self._review_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            row2, text="리뷰 모드 (포즈 확인 후 수동 승인)",
            variable=self._review_var, font=kf(12),
            fg_color=C_ACCENT, hover_color="#00cc66",
            border_color=C_BORDER, corner_radius=0,
            text_color=C_TEXT,
        ).pack(side="left")

        # 실행 버튼
        btn_row = ctk.CTkFrame(pad, fg_color="transparent", corner_radius=0)
        btn_row.pack(fill="x", pady=(0, 10))
        self._run_btn = _px_btn(btn_row, "[ RUN ] 가져오기 시작", self._run, 200, accent=True)
        self._run_btn.pack(side="left", padx=(0, 8))
        _px_btn(btn_row, "[ CLR ] 로그 지우기",
                lambda: self._log.delete("1.0", "end"), 160).pack(side="left")

        # 로그
        _px_label(pad, "[ LOG OUTPUT ]", 10, C_ACCENT, mono=True).pack(anchor="w", pady=(0, 4))
        self._log = _px_textbox(pad, height=320)
        self._log.pack(fill="both", expand=True)

    def on_activate(self): pass
    def on_deactivate(self): pass

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self._dir_var.get())
        if d:
            self._dir_var.set(d)

    def _run(self):
        if self._running:
            return
        self._running = True
        self._run_btn.configure(state="disabled", text="[ ... ] 실행 중")
        args = [sys.executable, "video_importer.py",
                "--videos_dir", self._dir_var.get()]
        if self._review_var.get():
            args.append("--review")
        self._log.insert("end", f">> {' '.join(args)}\n")
        self._start_subprocess(args)

    def _start_subprocess(self, args):
        def reader():
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            for line in proc.stdout:
                self._log_q.put(line.rstrip())
            proc.wait()
            self._log_q.put(f"\n>> [완료] exit={proc.returncode}")
            self._log_q.put("__DONE__")
        threading.Thread(target=reader, daemon=True).start()
        self._poll_log()

    def _poll_log(self):
        while not self._log_q.empty():
            line = self._log_q.get_nowait()
            if line == "__DONE__":
                self._running = False
                self._run_btn.configure(state="normal", text="[ RUN ] 가져오기 시작")
                return
            self._log.insert("end", line + "\n")
            self._log.see("end")
        self.after(100, self._poll_log)


# ═══════════════════════════════════════════════════════════════════════════════
# 학습 페이지
# ═══════════════════════════════════════════════════════════════════════════════
class TrainerPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C_BG, corner_radius=0)
        self.app = app
        self._log_q: queue.Queue = queue.Queue()
        self._running = False
        self._build()

    def _build(self):
        pad = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        pad.pack(fill="both", expand=True, padx=16, pady=12)

        _px_label(pad, "[ MODEL TRAINER ]", 13, C_ACCENT, mono=True).pack(anchor="w", pady=(0, 10))

        # 데이터셋 정보
        info_frame = _px_frame(pad)
        info_frame.pack(fill="x", pady=(0, 10))
        hdr = ctk.CTkFrame(info_frame, fg_color=C_SURFACE, corner_radius=0, height=24)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="[ DATASET INFO ]", font=pf(10),
                     text_color=C_ACCENT).pack(side="left", padx=6)
        self._refresh_btn = _px_btn(hdr, "새로고침", self._refresh_stats, 90)
        self._refresh_btn.pack(side="right", padx=4, pady=2)
        self._stats_lbl = ctk.CTkLabel(
            info_frame, text="", font=pf(11),
            text_color=C_TEXT, justify="left",
        )
        self._stats_lbl.pack(anchor="w", padx=10, pady=6)

        # 학습 버튼
        btn_row = ctk.CTkFrame(pad, fg_color="transparent", corner_radius=0)
        btn_row.pack(fill="x", pady=(0, 10))
        self._train_btn = _px_btn(btn_row, "[ TRAIN ] 학습 시작", self._run, 200, accent=True)
        self._train_btn.pack(side="left", padx=(0, 8))
        _px_btn(btn_row, "[ CLR ] 로그 지우기",
                lambda: self._log.delete("1.0", "end"), 160).pack(side="left", padx=(0, 8))
        _px_btn(btn_row, "[ IMG ] 그래프 열기", self._open_chart, 150).pack(side="left")

        # 로그
        _px_label(pad, "[ TRAINING LOG ]", 10, C_ACCENT, mono=True).pack(anchor="w", pady=(0, 4))
        self._log = _px_textbox(pad, height=300)
        self._log.pack(fill="both", expand=True)

    def on_activate(self):
        self._refresh_stats()

    def on_deactivate(self): pass

    def _refresh_stats(self):
        labels_path = os.path.join(DATA_DIR, 'labels.json')
        if not os.path.exists(labels_path):
            self._stats_lbl.configure(text="  데이터 없음 — video_importer 또는 data_collector를 먼저 실행하세요")
            return
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        lines = []
        total = 0
        for name in sorted(labels):
            d = os.path.join(DATA_DIR, name)
            n = len([f for f in os.listdir(d) if f.endswith('.npy')]) if os.path.isdir(d) else 0
            total += n
            status = "OK" if n >= 30 else "LOW"
            lines.append(f"  [{status:3s}]  {name:<20s}  {n:3d} seqs")

        lines.append(f"\n  TOTAL: {len(labels)} labels  |  {total} sequences")
        self._stats_lbl.configure(text="\n".join(lines))

    def _run(self):
        if self._running:
            return
        self._running = True
        self._train_btn.configure(state="disabled", text="[ ... ] 학습 중")
        args = [sys.executable, "train.py"]
        self._log.insert("end", f">> {' '.join(args)}\n\n")
        self._start_subprocess(args)

    def _start_subprocess(self, args):
        def reader():
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            for line in proc.stdout:
                self._log_q.put(line.rstrip())
            proc.wait()
            self._log_q.put(f"\n>> [완료] exit={proc.returncode}")
            self._log_q.put("__DONE__")
        threading.Thread(target=reader, daemon=True).start()
        self._poll_log()

    def _poll_log(self):
        while not self._log_q.empty():
            line = self._log_q.get_nowait()
            if line == "__DONE__":
                self._running = False
                self._train_btn.configure(state="normal", text="[ TRAIN ] 학습 시작")
                self._refresh_stats()
                return
            self._log.insert("end", line + "\n")
            self._log.see("end")
        self.after(100, self._poll_log)

    def _open_chart(self):
        chart = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'models', 'training_history.png')
        if not os.path.exists(chart):
            messagebox.showinfo("없음", "학습 그래프가 없습니다. 먼저 학습을 완료하세요.")
            return
        if platform.system() == "Darwin":
            subprocess.Popen(["open", chart])
        elif platform.system() == "Windows":
            os.startfile(chart)
        else:
            subprocess.Popen(["xdg-open", chart])


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 앱
# ═══════════════════════════════════════════════════════════════════════════════
_NAV_ITEMS = [
    ("▶  번역기",    "translator"),
    ("◉  데이터 수집", "collector"),
    ("↓  영상 가져오기", "importer"),
    ("⚙  학습",      "trainer"),
]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hand-Bridge  //  Korean Sign Language Translator")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(fg_color=C_BG)

        self._active_page: str | None = None
        self._nav_btns: dict[str, ctk.CTkButton] = {}
        self._pages: dict[str, ctk.CTkFrame] = {}

        self._build_header()
        self._build_body()
        self.show_page("translator")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=0,
                           height=38, border_width=1, border_color=C_BORDER)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="░░ Hand-Bridge  //  Real-time Korean Sign Language Translator ░░",
                     font=pf(11, bold=True), text_color=C_ACCENT).pack(side="left", padx=12)
        ctk.CTkLabel(hdr, text="[ CPU MODE ]",
                     font=pf(10), text_color=C_TEXT_DIM).pack(side="right", padx=12)

    # ── 사이드바 + 콘텐츠 ─────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True)

        # 사이드바
        sidebar = ctk.CTkFrame(body, fg_color=C_PANEL, corner_radius=0,
                               width=SIDEBAR_W, border_width=1, border_color=C_BORDER)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="[ MENU ]", font=pf(10, bold=True),
                     text_color=C_ACCENT).pack(pady=(14, 8))

        for label, key in _NAV_ITEMS:
            btn = ctk.CTkButton(
                sidebar, text=label, width=SIDEBAR_W - 16,
                font=kf(12), fg_color=C_BTN, hover_color=C_BTN_HOVER,
                text_color=C_TEXT, corner_radius=0,
                border_width=1, border_color=C_BORDER,
                command=lambda k=key: self.show_page(k),
            )
            btn.pack(pady=3, padx=8)
            self._nav_btns[key] = btn

        # 버전 정보 (하단)
        ctk.CTkLabel(sidebar, text="Hand-Bridge v1.0  //  CPU-only",
                     font=pf(9), text_color=C_TEXT_DIM).pack(side="bottom", pady=10)

        # 콘텐츠
        self._content = ctk.CTkFrame(body, fg_color=C_BG, corner_radius=0)
        self._content.pack(side="left", fill="both", expand=True)

        for key, cls in [
            ("translator", TranslatorPage),
            ("collector",  CollectorPage),
            ("importer",   ImporterPage),
            ("trainer",    TrainerPage),
        ]:
            page = cls(self._content, self)
            self._pages[key] = page

    # ── 페이지 전환 ───────────────────────────────────────────────────────────
    def show_page(self, key: str):
        if self._active_page == key:
            return

        # 이전 페이지 정리
        if self._active_page:
            self._pages[self._active_page].on_deactivate()
            self._pages[self._active_page].pack_forget()
            self._nav_btns[self._active_page].configure(
                fg_color=C_BTN, text_color=C_TEXT)

        self._active_page = key
        self._pages[key].pack(fill="both", expand=True)
        self._pages[key].on_activate()
        self._nav_btns[key].configure(
            fg_color=C_ACCENT, text_color=C_BG)

    def _on_close(self):
        if self._active_page:
            self._pages[self._active_page].on_deactivate()
        self.destroy()


# ── 진입점 ────────────────────────────────────────────────────────────────────
def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
