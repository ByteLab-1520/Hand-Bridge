"""
🎯 동작 인식 액션 자동 실행 가이드

이 시스템은 한국수어 동작을 실시간으로 인식하고,
학습된 동작을 감지하면 자동으로 프로그램을 열거나 키보드 단축키를 실행합니다.

═══════════════════════════════════════════════════════════════════

📋 파일 구조

1. gesture_actions.py
   - 메인 프로그램: 실시간 카메라에서 동작을 인식하고 액션 실행
   - 실행: python gesture_actions.py

2. gesture_config_gui.py
   - 설정 GUI: 동작을 액션에 매핑
   - 실행: python gesture_config_gui.py

3. gesture_config.json
   - 저장된 설정 파일 (자동 생성)
   - 동작 ↔ 액션 매핑

4. gesture_log.txt
   - 실행 로그 (자동 생성)

═══════════════════════════════════════════════════════════════════

🚀 시작하는 방법

## 단계 1: 모델 학습
먼저 수어 동작 데이터를 수집하고 모델을 학습해야 합니다.

  python data_collector.py    # 동작 데이터 수집
  python train.py              # 모델 학습

## 단계 2: 동작-액션 매핑 설정

  python gesture_config_gui.py

### GUI에서 설정하기:
  1. 인식할 동작 선택 (좌측 목록)
     - 관람, 그립다, 꿈, 만나다, 사랑합니다, 여자, 행복, etc.

  2. 액션 타입 선택 (우측):
     - 프로그램 열기: Safari, Chrome, Mail 등
     - 키보드 실행: cmd+c, cmd+v, F11 등
     - 명령어 실행: open -a Calendar, 등

  3. 액션 값 설정:
     - 프로그램명 또는 단축키 입력
     - 예: Safari, cmd+space, open -a Calculator

  4. 테스트 실행으로 동작 확인

  5. 저장 버튼으로 설정 저장

## 단계 3: 액션 자동 실행 시작

  python gesture_actions.py

### 실행 중 제어:
  - Q: 프로그램 종료
  - S: 설정 GUI 열기
  - G: 로그 파일 열기 (실행 기록 확인)

═══════════════════════════════════════════════════════════════════

💡 사용 예시

### 예시 1: Safari 열기
설정:
  동작: "관람"
  액션 타입: 프로그램 열기
  값: Safari

실행:
  gesture_actions.py 실행 중에 "관람" 동작을 하면
  → Safari 자동 실행

### 예시 2: 스크린샷 캡처
설정:
  동작: "여자"
  액션 타입: 키보드 실행
  값: cmd+shift+5

실행:
  gesture_actions.py 실행 중에 "여자" 동작을 하면
  → cmd+shift+5 단축키 자동 실행 (스크린샷 열림)

### 예시 3: 계산기 실행
설정:
  동작: "꿈"
  액션 타입: 명령어 실행
  값: open -a Calculator

실행:
  gesture_actions.py 실행 중에 "꿈" 동작을 하면
  → 계산기 앱 자동 실행

═══════════════════════════════════════════════════════════════════

🎨 지원되는 액션 타입

1️⃣ 프로그램 열기 (macOS)
   - Safari, Chrome, Firefox, Mail, Music
   - Finder, VS Code, Xcode, etc.
   - 커스텀 앱경로 입력 가능: /Applications/AppName.app

2️⃣ 키보드 단축키
   - cmd+c (복사), cmd+v (붙여넣기), cmd+x (잘라내기)
   - cmd+z (되돌리기), cmd+a (모두선택), cmd+s (저장)
   - cmd+space (Spotlight), cmd+tab (앱 전환)
   - cmd+shift+5 (스크린샷), F11 (전체화면), Return, Escape

3️⃣ 명령어 실행
   - open -a AppName          # 앱 열기
   - open ~/Documents         # 폴더 열기
   - open https://google.com  # 브라우저에서 URL 열기

═══════════════════════════════════════════════════════════════════

⚙️ gesture_config.json 예시

{
  "enabled": true,
  "actions": {
    "관람": {
      "type": "app",
      "value": "Safari"
    },
    "그립다": {
      "type": "key",
      "value": "cmd+c"
    },
    "꿈": {
      "type": "command",
      "value": "open -a Calculator"
    },
    "여자": {
      "type": "key",
      "value": "cmd+shift+5"
    }
  }
}

═══════════════════════════════════════════════════════════════════

📊 실시간 모니터링

gesture_actions.py 실행 중:

┌─────────────────────────────────────────┐
│ 🔴 활성 | FPS: 30                       │ ← 시스템 상태
│                                         │
│                마지막 동작: 관람         │ ← 인식된 동작
│                                         │
│ 신뢰도: 85%                             │ ← 예측 정확도
│ ██████████████░░░░░░                   │
│                                         │
│ ●●●●●●○○○○○○○○○○○○○○○             │ ← 수집된 프레임 수
│                                         │
│ Q: 종료 S: 설정 열기 G: 로그           │
└─────────────────────────────────────────┘

・신뢰도 바: 녹색(높음) ↔ 파란색(낮음)
・점들: 모아진 프레임 수를 나타냄 (30개 모으면 인식 시도)
・중복 방지: 같은 동작은 2초 뒤에 다시 실행 가능

═══════════════════════════════════════════════════════════════════

🐛 문제 해결

Q1: "ModuleNotFoundError" 에러가 나요
A1: 필수 패키지 설치
    pip install -r requirements.txt

Q2: 동작이 인식되지 않아요
A2: 신뢰도 확인
    - gesture_actions.py 창에서 신뢰도가 80% 이상인지 확인
    - 데이터 재수집: python data_collector.py

Q3: 모델이 없다고 나와요
A3: 먼저 train.py 실행
    python train.py

Q4: 카메라가 인식되지 않아요
A4: config.py에서 CAMERA_INDEX 확인
    - 0번 카메라가 일반적
    - 다른 번호로 시도: config.py의 CAMERA_INDEX = 1 로 변경

Q5: 액션이 실행되지 않아요
A5: 1. gesture_config_gui.py에서 설정 확인
    2. gesture_log.txt 로그 확인
    3. 신뢰도가 충분히 높은지 확인 (≥80%)

═══════════════════════════════════════════════════════════════════

🔐 보안 및 팁

・같은 동작은 2초 쿨다운 적용 (오작동 방지)
・민감한 동작은 신뢰도 높게 설정 권장
・로그는 gesture_log.txt에 자동 저장
・설정은 gesture_config.json에 저장 (JSON 형식)

═══════════════════════════════════════════════════════════════════

💬 기술 스택

・OpenCV: 카메라 처리
・MediaPipe: 손/신체 감지
・TensorFlow/Keras: LSTM 모델
・customtkinter: UI

═══════════════════════════════════════════════════════════════════

질문이나 문제가 있으신가요? gesture_log.txt를 확인하세요!
"""

if __name__ == "__main__":
    print(__doc__)
