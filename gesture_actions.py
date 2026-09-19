"""
실시간 동작 인식 + 액션 자동 실행 — Gesture Action Executor

실행: python gesture_actions.py

기능:
- 실시간 카메라에서 한국수어 동작 인식
- 설정된 동작을 감지하면 자동으로 액션 실행
- 프로그램 열기, 키보드 단축키, 명령어 실행 등 지원

설정: gesture_config.json에서 동작 ↔ 액션 매핑
GUI 설정: python gesture_config_gui.py
"""

import cv2
import numpy as np
import json
import os
import time
import collections
import threading
import subprocess
import tensorflow as tf
from datetime import datetime

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    SEQUENCE_LENGTH, NUM_FEATURES,
    PREDICTION_THRESHOLD, STABLE_FRAMES,
    MODEL_PATH, LABELS_PATH,
)
from utils import extract_landmarks, draw_landmarks, put_korean_text, HolisticDetector, open_camera
from model import make_inference_fn, run_inference

CONFIG_FILE = "gesture_config.json"
LOG_FILE = "gesture_log.txt"

# 기본 설정
DEFAULT_CONFIG = {
    "enabled": True,
    "actions": {},
}


class ActionExecutor(threading.Thread):
    """액션 실행 스레드 (메인 스레드 블로킹 방지)"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = collections.deque(maxlen=10)
        self.running = True
        
    def add_action(self, action_type, value, gesture_name):
        """액션 큐에 추가"""
        self.queue.append((action_type, value, gesture_name))
        
    def run(self):
        """액션 큐 처리"""
        while self.running:
            if self.queue:
                action_type, value, gesture_name = self.queue.popleft()
                try:
                    self.execute(action_type, value, gesture_name)
                except Exception as e:
                    log(f"[ERROR] {gesture_name} 액션 실행 실패: {e}")
            time.sleep(0.01)
            
    def execute(self, action_type, value, gesture_name):
        """액션 실행"""
        import platform
        
        log(f"[ACTION] {gesture_name} → {action_type}: {value}")
        
        if action_type == "app":
            # 앱 열기 (macOS)
            if platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", value])
            else:
                subprocess.Popen(value, shell=True)
                
        elif action_type == "key":
            # 키보드 실행
            if platform.system() == "Darwin":
                # macOS: osascript 사용
                try:
                    subprocess.run(["osascript", "-e", 
                                  f'tell application "System Events" to keystroke "{value}" using cmd down'],
                                 timeout=2)
                except:
                    pass
            else:
                # Windows/Linux: pyautogui 사용
                try:
                    import pyautogui
                    pyautogui.hotkey(*value.split('+'))
                except ImportError:
                    log("[WARN] pyautogui 미설치. 키보드 실행 불가")
                    
        elif action_type == "command":
            # 명령어 실행
            subprocess.Popen(value, shell=True)
            
    def stop(self):
        self.running = False


class GestureRecognizer:
    """한국수어 동작 인식"""
    
    def __init__(self, model, idx_to_label, config):
        self.model = model
        self.infer = make_inference_fn(model)
        self.idx_to_label = idx_to_label
        self.config = config
        
        self.sequence = []
        self.last_prediction = None
        self.stable_count = 0
        self.last_added_gesture = None
        self.last_hand_time = time.time()
        
        # 스무딩
        self.pred_queue = collections.deque(maxlen=STABLE_FRAMES)
        
        # 중복 방지 타이머
        self.gesture_cooldown = {}  # { gesture: last_time }
        self.cooldown_duration = 2.0  # 2초 내 같은 동작 재실행 금지
        
    def process_frame(self, landmarks, hand_detected=True):
        """프레임 처리 및 동작 인식
        
        Returns:
            (인식된_동작_이름, 신뢰도)
        """
        if not hand_detected:
            self.sequence.clear()
            if time.time() - self.last_hand_time > 3.0:
                self.pred_queue.clear()
            return None, 0.0
        
        self.last_hand_time = time.time()
        
        self.sequence.append(landmarks)
        if len(self.sequence) > SEQUENCE_LENGTH:
            self.sequence.pop(0)
            
        if len(self.sequence) < SEQUENCE_LENGTH:
            return None, 0.0
        
        # 모델 예측
        probs = run_inference(self.infer, self.sequence)
        conf = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
        pred_label = self.idx_to_label.get(pred_idx, '?')
        
        self.pred_queue.append((pred_label, conf))
        
        # STABLE_FRAMES 연속 같은 예측 검사
        if len(self.pred_queue) < STABLE_FRAMES:
            return None, conf
        
        labels_in_queue = [p[0] for p in self.pred_queue]
        confs_in_queue = [p[1] for p in self.pred_queue]
        
        if len(set(labels_in_queue)) == 1 and min(confs_in_queue) >= PREDICTION_THRESHOLD:
            stable_gesture = labels_in_queue[0]
            avg_conf = float(np.mean(confs_in_queue))
            
            # 중복 방지 체크
            now = time.time()
            if stable_gesture in self.gesture_cooldown:
                if now - self.gesture_cooldown[stable_gesture] < self.cooldown_duration:
                    return None, avg_conf  # 쿨다운 중
            
            # 새로운 동작 인식
            if stable_gesture != self.last_added_gesture:
                self.gesture_cooldown[stable_gesture] = now
                self.last_added_gesture = stable_gesture
                self.pred_queue.clear()
                self.sequence.clear()
                return stable_gesture, avg_conf
        
        return None, conf


def load_model_and_labels():
    """모델과 레이블 로드"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일 없음: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"레이블 파일 없음: {LABELS_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    idx_to_label = {v: k for k, v in label_map.items()}
    return model, idx_to_label


def load_config():
    """설정 파일 로드"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_CONFIG.copy()


def log(message):
    """로그 출력 및 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    except:
        pass


def draw_ui(frame, recognizer, current_conf, fps, config):
    """UI 오버레이 그리기"""
    h, w = frame.shape[:2]
    
    # ── 상단: 상태 표시 ────────────────────────────────────────────────
    status = "🔴 활성" if config.get("enabled") else "⚫ 비활성"
    frame = put_korean_text(
        frame, f"{status}  |  FPS: {fps:.0f}",
        (10, 25), font_size=24, color=(0, 255, 0) if config.get("enabled") else (100, 100, 100),
    )
    
    # ── 중앙: 마지막 인식된 동작 ────────────────────────────────────────
    if recognizer.last_added_gesture:
        frame = put_korean_text(
            frame, f"마지막 동작: {recognizer.last_added_gesture}",
            (w//2, h//2), font_size=28, color=(0, 255, 0), 
        )
    
    # ── 하단: 신뢰도 바 ────────────────────────────────────────────────
    bar_y = h - 60
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 20), (60, 60, 60), -1)
    bar_width = int((w - 20) * min(current_conf, 1.0))
    bar_color = (0, 220, 0) if current_conf >= PREDICTION_THRESHOLD else (0, 140, 255)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 20), bar_color, -1)
    
    frame = put_korean_text(
        frame, f"신뢰도: {current_conf:.0%}",
        (10, bar_y + 22), font_size=18, color=(200, 200, 200),
    )
    
    # ── 시퀀스 진행도 (작은 점들) ────────────────────────────────────
    filled = len(recognizer.sequence)
    dot_y = h - 80
    for i in range(SEQUENCE_LENGTH):
        color = (0, 255, 0) if i < filled else (80, 80, 80)
        cv2.circle(frame, (10 + i * (w - 20) // SEQUENCE_LENGTH, dot_y), 3, color, -1)
    
    # ── 좌측 하단: 제어 방법 ────────────────────────────────────────────
    frame = put_korean_text(
        frame, "Q: 종료  S: 설정 열기  G: 로그",
        (10, h - 24), font_size=16, color=(150, 150, 150),
    )
    
    return frame


def show_config_summary(config):
    """설정 요약 표시"""
    actions = config.get("actions", {})
    if not actions:
        log("[INFO] 설정된 동작-액션이 없습니다. gesture_config_gui.py를 실행하여 설정하세요.")
        return
    
    log("[INFO] ═══ 동작-액션 매핑 ═══")
    for gesture, action in actions.items():
        log(f"       {gesture:10} → {action.get('type', '?'):7} | {action.get('value', '?')}")
    log("[INFO] ══════════════════════")


def main():
    print("═" * 60)
    print("   실시간 동작 인식 + 액션 자동 실행")
    print("═" * 60)
    
    try:
        log("[INFO] 모델 로딩 중...")
        model, idx_to_label = load_model_and_labels()
        log(f"[INFO] 인식 가능한 동작: {list(idx_to_label.values())}")
        
        config = load_config()
        show_config_summary(config)
        
        if not config.get("enabled"):
            log("[WARN] 서비스가 비활성화되어 있습니다. gesture_config_gui.py에서 활성화하세요.")
        
    except Exception as e:
        log(f"[ERROR] 초기화 실패: {e}")
        return
    
    # 스레드 준비
    executor = ActionExecutor()
    executor.start()
    
    recognizer = GestureRecognizer(model, idx_to_label, config)
    detector = HolisticDetector()
    
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    
    if not cap.isOpened():
        log("[ERROR] 카메라를 열 수 없습니다.")
        return
    
    log("[INFO] 동작 인식 시작. 'Q'를 눌러 종료합니다.")
    
    fps_prev = time.time()
    fps = 0.0
    current_conf = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            
            # 랜드마크 추출
            landmarks = extract_landmarks(results)
            
            # 동작 인식
            gesture, current_conf = recognizer.process_frame(
                landmarks, 
                hand_detected=results.hand_detected
            )
            
            # 액션 실행
            if gesture and config.get("enabled"):
                actions = config.get("actions", {})
                if gesture in actions:
                    action = actions[gesture]
                    executor.add_action(
                        action.get("type", "command"),
                        action.get("value", ""),
                        gesture
                    )
            
            # 랜드마크 그리기 및 UI 렌더링
            frame = draw_landmarks(frame, results)
            frame = draw_ui(frame, recognizer, current_conf, fps, config)
            
            # FPS 계산
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_prev, 1e-6))
            fps_prev = now
            
            # 표시
            cv2.imshow('동작 인식 + 액션 실행', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                log("[INFO] 사용자 종료 명령")
                break
            elif key == ord('s'):
                log("[INFO] 설정 GUI 열기...")
                try:
                    subprocess.Popen(["/usr/local/bin/python3", "gesture_config_gui.py"])
                except:
                    log("[WARN] gesture_config_gui.py를 실행할 수 없습니다.")
            elif key == ord('g'):
                log("[INFO] 로그 파일 열기...")
                try:
                    subprocess.Popen(["open", LOG_FILE])
                except:
                    pass
    
    except KeyboardInterrupt:
        log("[INFO] 인터럽트 신호 받음")
    except Exception as e:
        log(f"[ERROR] 실행 중 오류: {e}")
    finally:
        executor.stop()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        log("[INFO] 종료 완료")
        print("═" * 60)


if __name__ == '__main__':
    main()
