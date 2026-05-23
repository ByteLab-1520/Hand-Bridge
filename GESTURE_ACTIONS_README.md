# 🎯 동작 인식 액션 자동 실행 시스템

한국수어 동작을 실시간으로 인식하고, 학습된 동작을 감지하면 자동으로 프로그램을 열거나 키보드 단축키를 실행하는 시스템입니다.

## 📋 프로그램 구성

| 파일 | 설명 |
|------|------|
| **gesture_actions.py** | 메인 프로그램: 실시간 동작 인식 및 액션 자동 실행 |
| **gesture_config_gui.py** | 설정 GUI: 동작 ↔ 액션 매핑 |
| **gesture_config.json** | 저장된 설정 파일 (자동 생성) |
| **gesture_log.txt** | 실행 로그 (자동 생성) |
| **GESTURE_ACTIONS_GUIDE.py** | 상세 가이드 (python GESTURE_ACTIONS_GUIDE.py) |

## 🚀 빠른 시작

### 1단계: 동작 학습 (최초 1회만)
```bash
# 동작 데이터 수집
python data_collector.py

# 모델 학습
python train.py
```

### 2단계: 동작-액션 매핑 설정
```bash
python gesture_config_gui.py
```

GUI에서:
- 인식할 동작 선택 (관람, 그립다, 꿈, 만나다, 사랑합니다, 여자, 행복 등)
- 액션 타입 선택 (프로그램/키보드/명령어)
- 액션 값 설정 및 테스트
- 저장 버튼 클릭

### 3단계: 액션 자동 실행 시작
```bash
python gesture_actions.py
```

실행 중 제어:
- **Q**: 프로그램 종료
- **S**: 설정 GUI 열기
- **G**: 로그 파일 열기

## 💡 사용 예시

### 예시 1: Safari 자동 열기
```
설정:
  동작: "관람"
  액션: 프로그램 열기 → Safari

사용:
  "관람" 수어를 하면 Safari가 자동으로 실행됨
```

### 예시 2: 스크린샷 캡처
```
설정:
  동작: "여자"
  액션: 키보드 실행 → cmd+shift+5

사용:
  "여자" 수어를 하면 스크린샷 도구가 실행됨
```

### 예시 3: 계산기 실행
```
설정:
  동작: "꿈"
  액션: 명령어 실행 → open -a Calculator

사용:
  "꿈" 수어를 하면 계산기가 실행됨
```

## 🎨 지원되는 액션

### 1️⃣ 프로그램 열기 (macOS)
- Safari, Chrome, Firefox, Mail, Music, Finder
- VS Code, Xcode, Slack, Discord 등
- 커스텀 경로: `/Applications/AppName.app`

### 2️⃣ 키보드 단축키
- `cmd+c` 복사 / `cmd+v` 붙여넣기
- `cmd+z` 되돌리기 / `cmd+a` 모두선택
- `cmd+space` Spotlight / `cmd+tab` 앱 전환
- `cmd+shift+5` 스크린샷 / `F11` 전체화면

### 3️⃣ 명령어 실행
- `open -a AppName` - 앱 열기
- `open ~/Documents` - 폴더 열기
- `open https://google.com` - 웹페이지 열기

## ⚙️ 설정 파일 예시 (gesture_config.json)

```json
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
```

## 📊 실시간 모니터링

```
┌─────────────────────────────────────────┐
│ 🔴 활성 | FPS: 30                       │ ← 시스템 상태
│                                         │
│                마지막 동작: 관람         │ ← 인식된 동작
│                                         │
│ 신뢰도: 85%                             │ ← 예측 정확도
│ ██████████████░░░░░░                   │
│                                         │
│ ●●●●●●○○○○○○○○○○○○○○○             │ ← 수집 프레임
│                                         │
│ Q: 종료 S: 설정 G: 로그                 │
└─────────────────────────────────────────┘
```

## 🔧 주요 기능

- ✅ **실시간 인식**: 초당 30FPS 이상 처리
- ✅ **중복 방지**: 같은 동작은 2초 쿨다운 적용
- ✅ **신뢰도 필터**: 80% 이상 신뢰도에서만 인식
- ✅ **배경 스레드**: 액션 실행이 인식을 방해하지 않음
- ✅ **자동 로깅**: 모든 인식과 액션이 로그에 기록됨
- ✅ **GUI 설정**: 복잡한 코드 수정 없이 GUI로 설정

## 🐛 문제 해결

| 문제 | 해결책 |
|------|--------|
| ModuleNotFoundError | `pip install -r requirements.txt` 실행 |
| 동작이 인식 안 됨 | data_collector.py로 데이터 재수집 및 train.py 재학습 |
| 모델 파일 없음 | train.py 실행하여 모델 생성 |
| 카메라 인식 안 됨 | config.py의 CAMERA_INDEX 값 변경 |
| 액션 실행 안 됨 | gesture_log.txt 확인하여 신뢰도/설정 확인 |

## 💬 로그 확인

실행 중 **G 키**를 누르거나 직접 확인:
```bash
cat gesture_log.txt
```

로그 예시:
```
[2024-05-15 10:30:45] [INFO] 모델 로딩 중...
[2024-05-15 10:30:48] [INFO] 인식 가능한 동작: ['관람', '그립다', '꿈', '만나다', ...]
[2024-05-15 10:31:02] [ACTION] 관람 → app: Safari
[2024-05-15 10:31:15] [ACTION] 여자 → key: cmd+shift+5
```

## 🔒 보안 및 팁

- 같은 동작은 2초 쿨다운으로 오작동 방지
- 민감한 동작은 높은 신뢰도로 설정 권장
- 설정은 JSON 파일로 관리되므로 백업 가능
- 로그는 자동 저장되어 문제 추적 가능

## 📚 상세 가이드

더 자세한 내용은:
```bash
python GESTURE_ACTIONS_GUIDE.py
```

## 🛠️ 기술 스택

- **OpenCV** - 카메라 처리
- **MediaPipe** - 손/신체 감지
- **TensorFlow/Keras** - LSTM 동작 인식 모델
- **customtkinter** - 모던 UI
- **Python 3.10+** - 프로그래밍 언어

---

**문제가 있으신가요?** `gesture_log.txt` 파일을 확인하거나 설정을 다시 검토해보세요!
