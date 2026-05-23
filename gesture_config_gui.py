"""
동작 액션 설정 GUI — Gesture Action Configuration

실행: python gesture_config_gui.py

기능:
- 학습된 수어 동작을 선택
- 각 동작에 액션 설정 (프로그램 열기, 키보드 단축키 실행 등)
- 설정을 gesture_config.json에 저장
"""

import json
import os
import customtkinter as ctk
from tkinter import messagebox, simpledialog
import platform

CONFIG_FILE = "gesture_config.json"

# 기본 설정 구조
DEFAULT_CONFIG = {
    "actions": {},  # { "gesture_name": {"type": "app|key|command", "value": "..."} }
    "enabled": True,
}

GESTURE_LABELS = [
    "관람",  # view/watch
    "그립다",  # miss
    "꿈",   # dream
    "만나다",  # meet
    "사랑합니다",  # love
    "여자",  # woman
    "행복",  # happy
]

ACTION_TYPES = {
    "app": "프로그램 열기",
    "key": "키보드 실행",
    "command": "명령어 실행",
}

# macOS 기본 앱
MACOS_APPS = {
    "Safari": "/Applications/Safari.app",
    "Chrome": "/Applications/Google Chrome.app",
    "Firefox": "/Applications/Firefox.app",
    "Mail": "/Applications/Mail.app",
    "Music": "/Applications/Music.app",
    "Finder": "/Applications/Finder.app",
    "Spotlight": "cmd+space",  # 특수 키
    "Screenshot": "cmd+shift+5",
    "스크린샷": "cmd+shift+5",
}

KEYBOARD_SHORTCUTS = [
    "cmd+c",  # Copy
    "cmd+v",  # Paste
    "cmd+x",  # Cut
    "cmd+z",  # Undo
    "cmd+a",  # Select All
    "cmd+s",  # Save
    "cmd+space",  # Spotlight
    "cmd+tab",  # Switch App
    "cmd+shift+5",  # Screenshot
    "F11",  # Fullscreen
    "Return",  # Enter
    "Escape",  # Esc
]


class GestureConfigGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("동작 액션 설정")
        self.geometry("800x600")
        self.config = self.load_config()
        
        self.setup_ui()
        
    def load_config(self):
        """기존 설정 불러오기"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """설정 저장"""
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("저장", "설정이 저장되었습니다!")
        
    def setup_ui(self):
        """UI 레이아웃"""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # ── Header ────────────────────────────────────────────────────
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(header, text="🎯 동작 액션 설정", font=("Arial", 20, "bold")).pack(side="left")
        
        self.enabled_var = ctk.BooleanVar(value=self.config.get("enabled", True))
        enable_btn = ctk.CTkSwitch(
            header, text="활성화",
            variable=self.enabled_var,
            command=self.toggle_enabled,
            onvalue=True, offvalue=False
        )
        enable_btn.pack(side="right")
        self._enable_switch = enable_btn
        
        # ── Main Content ──────────────────────────────────────────────
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 좌측: 동작 목록 (버튼으로 변경)
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(left_frame, text="인식 가능한 동작", font=("Arial", 14, "bold")).pack()
        
        # 스크롤 가능한 프레임
        scrollable_frame = ctk.CTkScrollableFrame(left_frame, width=150)
        scrollable_frame.pack(fill="both", expand=True, pady=10)
        
        self._gesture_buttons = {}
        for gesture in GESTURE_LABELS:
            status = "✓" if gesture in self.config.get("actions", {}) else "○"
            btn = ctk.CTkButton(
                scrollable_frame,
                text=f"{status} {gesture}",
                command=lambda g=gesture: self.on_gesture_select(g),
                fg_color="#2a3a55",
                hover_color="#00cc6a",
                width=140
            )
            btn.pack(pady=4, padx=5, fill="x")
            self._gesture_buttons[gesture] = btn
        
        # 우측: 액션 설정
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        ctk.CTkLabel(right_frame, text="액션 설정", font=("Arial", 14, "bold")).pack()
        
        # 선택된 동작 표시
        self.gesture_label = ctk.CTkLabel(right_frame, text="동작을 선택하세요", 
                                         font=("Arial", 12), text_color="cyan")
        self.gesture_label.pack(pady=10)
        
        # 액션 타입 선택
        ctk.CTkLabel(right_frame, text="액션 타입", font=("Arial", 11, "bold")).pack()
        self.action_type_var = ctk.StringVar(value="app")
        
        type_frame = ctk.CTkFrame(right_frame)
        type_frame.pack(fill="x", pady=5)
        
        for action_type, label in ACTION_TYPES.items():
            ctk.CTkRadioButton(
                type_frame, text=label, variable=self.action_type_var,
                value=action_type, command=self.on_action_type_changed
            ).pack(anchor="w", pady=3)
        
        # 액션 값 입력
        ctk.CTkLabel(right_frame, text="액션 값", font=("Arial", 11, "bold")).pack(pady=(15, 5))
        
        self.value_frame = ctk.CTkFrame(right_frame)
        self.value_frame.pack(fill="both", expand=True, pady=5)
        
        self.on_action_type_changed()
        
        # ── Buttons ───────────────────────────────────────────────────
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkButton(button_frame, text="저장", command=self.save_config,
                     fg_color="green", text_color="white").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="삭제", command=self.delete_action,
                     fg_color="red", text_color="white").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="테스트 실행", command=self.test_action,
                     fg_color="blue", text_color="white").pack(side="left", padx=5)
        
        self._selected_gesture = None
        
    def toggle_enabled(self):
        """서비스 활성화/비활성화"""
        self.config["enabled"] = self.enabled_var.get()
        self.save_config()
        
    def on_gesture_select(self, gesture):
        """동작 선택"""
        if gesture in GESTURE_LABELS:
            self._selected_gesture = gesture
            self.gesture_label.configure(text=f"선택: {gesture}")
            self.load_gesture_config(gesture)
                
    def load_gesture_config(self, gesture):
        """해당 동작의 액션 설정 불러오기"""
        actions = self.config.get("actions", {})
        if gesture in actions:
            action = actions[gesture]
            self.action_type_var.set(action.get("type", "app"))
            self.on_action_type_changed()
            
            if "value" in action:
                if action["type"] == "app":
                    self.app_dropdown.set(action["value"])
                else:
                    self.value_input.delete(0, "end")
                    self.value_input.insert(0, action["value"])
                    
    def on_action_type_changed(self):
        """액션 타입 변경시 입력 필드 업데이트"""
        # 기존 위젯 제거
        for widget in self.value_frame.winfo_children():
            widget.destroy()
            
        action_type = self.action_type_var.get()
        
        if action_type == "app":
            # 앱 드롭다운
            self.app_dropdown = ctk.CTkComboBox(
                self.value_frame, values=list(MACOS_APPS.keys())
            )
            self.app_dropdown.pack(fill="x", pady=5)
            
        elif action_type == "key":
            # 키보드 단축키
            ctk.CTkLabel(self.value_frame, text="단축키 또는 키 이름", 
                        font=("Arial", 10)).pack(anchor="w", pady=5)
            
            self.value_input = ctk.CTkEntry(self.value_frame, 
                                           placeholder_text="예: cmd+c, F11")
            self.value_input.pack(fill="x", pady=5)
            
            # 추천 단축키
            ctk.CTkLabel(self.value_frame, text="추천 단축키:", 
                        font=("Arial", 9)).pack(anchor="w", pady=(10, 5))
            
            shortcuts_str = ", ".join(KEYBOARD_SHORTCUTS[:5]) + "..."
            ctk.CTkLabel(self.value_frame, text=shortcuts_str, 
                        font=("Arial", 8), text_color="gray").pack(anchor="w")
                        
        else:  # command
            # 명령어
            ctk.CTkLabel(self.value_frame, text="실행할 명령어", 
                        font=("Arial", 10)).pack(anchor="w", pady=5)
            
            self.value_input = ctk.CTkEntry(self.value_frame, 
                                           placeholder_text="예: open -a Calculator")
            self.value_input.pack(fill="x", pady=5)
            
            ctk.CTkLabel(self.value_frame, text="💡 macOS 예: open -a 앱이름", 
                        font=("Arial", 8), text_color="gray").pack(anchor="w")
    
    def delete_action(self):
        """선택된 동작의 액션 삭제"""
        if not self._selected_gesture:
            messagebox.showwarning("경고", "동작을 선택해주세요")
            return
            
        if self._selected_gesture in self.config.get("actions", {}):
            del self.config["actions"][self._selected_gesture]
            self.save_config()
            messagebox.showinfo("삭제", f"'{self._selected_gesture}' 액션이 삭제되었습니다")
            
            # 버튼 텍스트 업데이트
            if self._selected_gesture in self._gesture_buttons:
                self._gesture_buttons[self._selected_gesture].configure(
                    text=f"○ {self._selected_gesture}"
                )
            
            # 입력 필드 초기화
            self.gesture_label.configure(text="동작을 선택하세요")
            self._selected_gesture = None
            self.action_type_var.set("app")
            self.on_action_type_changed()
            
    def test_action(self):
        """액션 테스트 실행"""
        if not self._selected_gesture:
            messagebox.showwarning("경고", "동작을 선택해주세요")
            return
            
        action_type = self.action_type_var.get()
        
        if action_type == "app":
            if not hasattr(self, 'app_dropdown'):
                messagebox.showwarning("경고", "앱을 선택해주세요")
                return
            value = self.app_dropdown.get()
        else:
            if not hasattr(self, 'value_input'):
                messagebox.showwarning("경고", "값을 입력해주세요")
                return
            value = self.value_input.get()
        
        if not value:
            messagebox.showwarning("경고", "값을 입력해주세요")
            return
        
        # 액션 저장
        self.config.setdefault("actions", {})[self._selected_gesture] = {
            "type": action_type,
            "value": value
        }
        self.save_config()
        
        # 버튼 텍스트 업데이트 (✓ 표시)
        if self._selected_gesture in self._gesture_buttons:
            self._gesture_buttons[self._selected_gesture].configure(
                text=f"✓ {self._selected_gesture}"
            )
        
        # 테스트 실행
        try:
            self.execute_action(action_type, value)
            messagebox.showinfo("성공", f"액션이 실행되었습니다!\n\n동작: {self._selected_gesture}\n액션: {value}")
        except Exception as e:
            messagebox.showerror("오류", f"액션 실행 중 오류: {e}")
    
    @staticmethod
    def execute_action(action_type, value):
        """액션 실행"""
        import subprocess
        
        if action_type == "app":
            # 앱 열기
            app_path = MACOS_APPS.get(value, value)
            if app_path.startswith("cmd+") or app_path.startswith("shift+"):
                # 키보드 단축키는 제어할 수 없음
                raise Exception(f"앱 실행 불가: {value}")
            else:
                subprocess.Popen(["open", "-a", value])
                
        elif action_type == "key":
            # 키보드 실행 (macOS용 - osascript 사용)
            script = f"""
            tell application "System Events"
                keystroke "{value}" using cmd down
            end tell
            """
            subprocess.run(["osascript", "-e", script])
            
        elif action_type == "command":
            # 명령어 실행
            subprocess.Popen(value, shell=True)


if __name__ == "__main__":
    app = GestureConfigGUI()
    app.mainloop()
