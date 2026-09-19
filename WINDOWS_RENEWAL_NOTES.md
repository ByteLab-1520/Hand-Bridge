# Hand-Bridge Windows 실행 안정화 및 성능 개선 기록

이 문서는 기존 Hand-Bridge 프로젝트를 Windows 발표 환경에서 점검하면서 발견한 문제와 수정 내용을 상세히 기록한다.

- 기존 기준 커밋: `8aec43c` (`Add Pause Function and Fix recognition rate.`)
- Windows 개선 커밋: `9abb6f1` (`Improve Windows setup and real-time performance`)
- 비교 범위: `8aec43c..9abb6f1`
- 주요 목표: Windows 설치 재현성, 웹캠 시작 안정성, 실시간 추론 성능, 발표 전 사전점검

> 이 개선은 프로그램 실행 안정성과 처리 속도를 대상으로 한다. 실제 수어 인식 정확도는 별도의 독립 테스트 데이터가 없어 보장하지 않는다.

---

## 1. 요약

기존 버전은 README에 Windows 지원이 표시되어 있었지만, 새 Windows 환경에서 안내된 설치 절차를 그대로 실행하면 TensorFlow가 설치되지 않는 문제가 있었다. TensorFlow를 수동 설치한 뒤에도 GUI 백그라운드 스레드에서 카메라가 프레임을 전달하지 않는 문제가 재현됐으며, 프레임마다 호출되는 Keras `Model.predict()`의 오버헤드도 실시간 처리 속도를 크게 제한했다.

이번 개선에서 다음 사항을 수정했다.

1. Windows에서 TensorFlow가 누락되지 않도록 패키지 조건 수정
2. 발표용으로 검증된 Windows 패키지 버전 목록 추가
3. Windows 설치 자동화 PowerShell 스크립트 추가
4. 콘솔 없이 GUI를 실행하는 배치 파일 추가
5. Windows 카메라 백엔드를 DirectShow 우선으로 변경
6. 여러 실행 파일이 동일한 카메라 초기화 함수를 사용하도록 통합
7. `Model.predict()`를 `tf.function` 기반 추론으로 교체
8. MediaPipe 초기화 전에 카메라 미리보기를 먼저 표시
9. TensorFlow 모델을 읽는 동안 로딩 화면 표시
10. 모델·레이블·카메라·MediaPipe를 한 번에 검사하는 사전점검 도구 추가
11. README의 잘못된 저장소 주소 및 Windows 실행 안내 수정
12. 가상환경, 캐시, 실행 결과물이 Git에 포함되지 않도록 `.gitignore` 추가

---

## 2. 검증 환경

실측 검증에 사용한 환경은 다음과 같다.

| 항목 | 값 |
|---|---|
| 운영체제 | Windows 10 22H2, 빌드 19045 |
| CPU 아키텍처 표기 | `AMD64` |
| Python | 3.12.14, 64-bit |
| TensorFlow CPU | 2.21.0 |
| MediaPipe | 1.0.1 |
| OpenCV | 5.0.0.93 |
| NumPy | 2.5.3 |
| 카메라 요청 해상도 | 1280×720 |
| Keras 모델 입력 | `(None, 30, 249)` |
| Keras 모델 출력 | `(None, 2)` |
| 포함된 레이블 | `만나다`, `안녕` |

---

## 3. 문제 1: Windows에서 TensorFlow가 설치되지 않음

### 기존 상태

기존 `requirements.txt`는 CPU용 TensorFlow 설치 조건으로 다음 값을 사용했다.

```text
tensorflow-cpu>=2.13.0; platform_machine == "x86_64"
tensorflow>=2.13.0;     platform_machine == "arm64"
```

그러나 Windows 64-bit 환경에서 Python의 `platform.machine()`은 일반적으로 `x86_64`가 아니라 `AMD64`를 반환한다. 이 때문에 `pip install -r requirements.txt` 실행 시 다음과 같이 TensorFlow 패키지가 조건 불일치로 제외됐다.

```text
Ignoring tensorflow-cpu: markers 'platform_machine == "x86_64"' don't match your environment
```

설치는 성공으로 종료되지만 TensorFlow가 실제로 설치되지 않으므로, 이후 모델을 불러올 때 `ModuleNotFoundError`가 발생한다. 사용자 입장에서는 설치가 성공한 것처럼 보인다는 점에서 특히 위험한 문제였다.

### 수정 내용

CPU 아키텍처 문자열 대신 운영체제를 기준으로 패키지를 선택하도록 변경했다.

```text
tensorflow-cpu>=2.13.0; platform_system == "Windows" or platform_system == "Linux"
tensorflow>=2.13.0;     platform_system == "Darwin"
```

Windows와 Linux에서는 `tensorflow-cpu`, macOS에서는 `tensorflow`를 설치한다. 이 조건은 Windows의 `AMD64` 표기와 무관하게 동작한다.

### 추가 수정

기존 `opencv-python`을 `opencv-contrib-python`으로 변경했다. 현재 MediaPipe 배포판이 contrib 빌드를 함께 사용하므로 두 OpenCV 배포판이 같은 `cv2` 파일을 중복 설치할 가능성을 줄이기 위한 조치다.

### 관련 파일

- `requirements.txt`
- `requirements-windows.txt`

---

## 4. 문제 2: 발표일에 패키지 버전이 달라질 수 있음

### 기존 상태

기존 의존성은 대부분 최소 버전만 지정했다.

```text
mediapipe>=0.10.0
numpy>=1.24.0
tensorflow-cpu>=2.13.0
```

이 방식은 설치 시점에 따라 서로 다른 최신 버전이 설치된다. 개발 PC에서는 실행됐더라도 발표 PC에서 더 새로운 패키지가 설치되면서 API 또는 모델 호환 문제가 생길 수 있다.

### 수정 내용

실제로 Windows에서 모델·카메라·GUI 검증에 사용한 직접 의존성 버전을 `requirements-windows.txt`에 고정했다.

```text
mediapipe==1.0.1
opencv-contrib-python==5.0.0.93
numpy==2.5.3
tensorflow-cpu==2.21.0
Pillow==12.3.0
scikit-learn==1.9.1
matplotlib==3.11.2
customtkinter==6.0.0
```

일반 개발 환경은 `requirements.txt`, Windows 발표 환경은 `requirements-windows.txt`를 사용하도록 분리했다.

### 관련 파일

- `requirements-windows.txt`
- `setup_windows.ps1`

---

## 5. 문제 3: Windows GUI에서 카메라 프레임이 나오지 않음

### 재현된 현상

OpenCV를 메인 스레드에서 실행한 단독 테스트에서는 카메라가 정상적으로 열렸다.

```text
camera_open=True
frame_read=True
shape=(480, 640, 3)
backend=MSMF
```

하지만 GUI의 `CameraWorker` 백그라운드 스레드에서는 20초 이상 다음 상태가 유지됐다.

```text
NO SIGNAL
00 FPS
```

### 원인

기존 코드는 다음과 같이 기본 백엔드로 카메라를 열었다.

```python
cv2.VideoCapture(CAMERA_INDEX)
```

Windows에서는 기본적으로 Media Foundation(`MSMF`)이 선택될 수 있다. 해당 환경에서는 MSMF 카메라 초기화가 GUI 백그라운드 스레드에서 멈추는 현상이 재현됐다.

### 수정 내용

`utils.py`에 공통 `open_camera()` 함수를 추가했다.

- Windows: DirectShow(`CAP_DSHOW`) 우선
- DirectShow 실패 시 Media Foundation(`CAP_MSMF`) 재시도
- 다른 운영체제: 기본 백엔드(`CAP_ANY`)
- 요청 해상도 통합 설정
- 캡처 버퍼 크기를 1로 설정해 오래된 프레임 누적 최소화
- 모든 백엔드가 실패하면 OpenCV 기본 선택 방식으로 마지막 재시도

이 함수는 다음 실행 경로에서 공통으로 사용한다.

- `gui.py`
- `translator.py`
- `gesture_actions.py`
- `data_collector.py`
- `debug.py`
- `preflight_windows.py`

### 수정 후 실측

DirectShow를 사용한 백그라운드 스레드 테스트 결과:

```text
카메라 열기: 약 1.46초
opened=True
read=True
```

GUI 20초 연속 스모크 테스트 결과:

```text
GUI 생성: 약 2.94초
첫 카메라 프레임: 약 2.16초
최종 표시 FPS: 약 19 FPS
모델 상태: MODEL OK | 2 labels
```

FPS는 카메라, MediaPipe 손 추적, 랜드마크 그리기, 이미지 크기 조정, Tkinter 표시가 포함된 GUI 기준이다. PC 성능과 카메라 드라이버에 따라 달라질 수 있다.

### 관련 파일

- `utils.py`
- `gui.py`
- `translator.py`
- `gesture_actions.py`
- `data_collector.py`
- `debug.py`
- `preflight_windows.py`

---

## 6. 문제 4: 모델 추론 호출이 실시간 처리 병목이 됨

### 기존 상태

기존 코드는 30프레임 시퀀스가 준비된 뒤 매 프레임마다 다음 코드를 실행했다.

```python
probs = model.predict(seq_input, verbose=0)[0]
```

`Model.predict()`는 대량 데이터 처리에 유용하지만, 실시간 루프에서 샘플 하나만 반복 추론할 때도 내부 입력 파이프라인 준비 비용이 발생한다.

### 기존 방식 실측

더미 입력 `(1, 30, 249)`을 사용한 반복 측정:

```text
평균 추론 시간: 약 155.67 ms
초당 추론 횟수: 약 6.42 calls/s
```

이 수치는 카메라 및 MediaPipe 시간을 포함하지 않은 모델 추론만의 측정값이다. 실제 GUI에서는 더 낮은 전체 프레임 속도가 나올 수 있다.

### 수정 내용

`model.py`에 두 함수를 추가했다.

#### `make_inference_fn(model)`

- 입력 크기 `(None, 30, 249)`와 `float32` 타입을 `TensorSpec`으로 고정
- `tf.function`으로 추론 그래프 생성
- 모델 로딩 시 `get_concrete_function()`을 호출해 첫 실제 수어 입력 전에 그래프 추적 완료
- `training=False`로 Dropout 등 학습 전용 동작 비활성화

#### `run_inference(infer, sequence)`

- 랜드마크 시퀀스를 `float32` NumPy 배열로 변환
- 배치 차원을 추가
- 미리 생성한 추론 그래프 호출
- 결과를 1차원 확률 벡터로 반환

다음 실행 경로가 공통 추론 함수를 사용하도록 변경됐다.

- `translator.py`
- `gesture_actions.py`
- `gui.py`

### 수정 후 실측

동일 입력에 대한 `tf.function` 반복 측정:

```text
평균 추론 시간: 약 7.11 ms
초당 추론 횟수: 약 140.64 calls/s
```

Windows 사전점검 전체 실행 중 측정에서는 다음 값이 확인됐다.

```text
최적화 추론: 약 167.8 calls/s
```

측정 시점과 CPU 부하에 따라 결과는 달라지므로, 개선 후 수치는 약 140~168 calls/s 범위로 해석하는 것이 적절하다.

### 수치 동일성 검증

무작위 `float32` 입력 5개에 대해 기존 `Model.predict()` 결과와 새 `tf.function` 결과를 비교했다.

```text
max_abs_difference=0
inference_equivalence=PASS
```

따라서 이번 변경은 모델의 예측 결과를 바꾸지 않고 호출 오버헤드만 줄였다.

### 관련 파일

- `model.py`
- `gui.py`
- `translator.py`
- `gesture_actions.py`

---

## 7. 문제 5: 앱 시작 중 아무 반응이 없는 것처럼 보임

### 기존 상태

GUI 페이지 생성 과정에서 TensorFlow와 Keras 모델을 동기적으로 불러왔다. 모델을 읽는 동안 메인 UI가 완전히 표시되지 않아, 사용자가 실행 실패로 오해할 수 있었다.

MediaPipe 초기화 역시 카메라 워커 안에서 먼저 수행됐기 때문에, 초기화가 끝날 때까지 카메라 영역에는 `NO SIGNAL`만 표시됐다.

### 수정 내용

1. 메인 앱 생성 직후 `모델을 불러오는 중입니다...` 안내 표시
2. `self.update()`를 호출해 모델 로딩 전에 실제 화면 렌더링
3. 카메라를 먼저 열고 초기 프레임을 GUI 큐로 전송
4. 그 다음 MediaPipe `HolisticDetector` 초기화

이제 모델 및 MediaPipe 초기화 중에도 앱이 시작됐다는 사실과 카메라 입력 여부를 더 빠르게 확인할 수 있다.

### 관련 파일

- `gui.py`

---

## 8. Windows 설치 자동화

### `setup_windows.ps1`

다음 작업을 자동으로 처리한다.

1. 프로젝트 폴더로 작업 위치 변경
2. Python Launcher의 3.12, 3.11, 3.10 순서로 호환 Python 검색
3. Launcher가 없으면 `python` 명령 검사
4. Microsoft Store 연결용 가짜 `python.exe` 별칭을 실제 Python으로 오인하지 않도록 실행 검사
5. `.venv` 가상환경 생성
6. `pip` 업그레이드
7. `requirements-windows.txt` 설치
8. OpenCV, MediaPipe, TensorFlow import 및 버전 출력

Windows PowerShell 5는 외부 명령의 표준 오류 출력을 `NativeCommandError`로 바꿀 수 있다. 이 때문에 Python 후보 검사 구간에서만 오류 처리 수준을 일시적으로 낮추고, 검사 후 기존 설정으로 복원한다.

실행 명령:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

### `run_windows.bat`

다음 작업을 처리한다.

- 스크립트 파일이 있는 프로젝트 폴더로 이동
- `.venv\Scripts\pythonw.exe` 존재 여부 확인
- UTF-8 관련 환경 변수 설정
- TensorFlow 일반 로그 최소화
- 별도 콘솔 창 없이 `gui.py` 실행

실행 방법:

```text
run_windows.bat 더블클릭
```

### 관련 파일

- `setup_windows.ps1`
- `run_windows.bat`
- `requirements-windows.txt`

---

## 9. 발표 전 사전점검 도구

`preflight_windows.py`를 새로 추가했다. 발표 PC에서 다음 요소를 한 번에 검사한다.

1. Python 및 운영체제 정보 출력
2. TensorFlow, MediaPipe, OpenCV 버전 출력
3. `labels.json` 로딩
4. Keras 모델 로딩
5. 모델 입력 형태가 `(None, 30, 249)`인지 확인
6. 모델 출력 개수와 레이블 개수가 같은지 확인
7. 최적화된 추론 함수 생성 및 워밍업
8. 더미 입력 50회 추론 속도 측정
9. 출력 확률의 형태와 합이 정상인지 확인
10. 실제 카메라 열기 및 한 프레임 읽기
11. 실제 캡처 해상도와 OpenCV 백엔드 출력
12. MediaPipe로 프레임 한 장 처리
13. 랜드마크 벡터 형태가 `(249,)`인지 확인

실행 명령:

```powershell
.\.venv\Scripts\python.exe .\preflight_windows.py
```

실제 검증 결과:

```text
[PASS] 모델: (None, 30, 249) -> (None, 2), 2 labels
[PASS] 최적화 추론: 167.8 calls/s
[PASS] 카메라: 1280x720
[PASS] MediaPipe: 47.0 ms, landmarks=(249,)
모든 필수 검사를 통과했습니다.
```

### 관련 파일

- `preflight_windows.py`

---

## 10. README 및 저장소 정리

### README 수정

기존 clone 예시는 실제 저장소가 아닌 자리표시자 주소를 사용했다.

```text
https://github.com/your-username/hand-bridge.git
```

이를 실제 저장소 주소로 변경했다.

```text
https://github.com/ByteLab-1520/Hand-Bridge.git
```

Windows 발표 환경을 위한 Python 설치, PowerShell 설정, 배치 파일 실행, 사전점검 명령도 추가했다.

### `.gitignore` 추가

다음 로컬 실행 파일이 Git에 들어가지 않도록 했다.

```text
.venv/
__pycache__/
*.py[cod]
output.txt
hand_bridge.log
```

---

## 11. 파일별 변경 목록

| 파일 | 변경 내용 |
|---|---|
| `.gitignore` | 가상환경, Python 캐시, 출력 및 로그 파일 제외 |
| `README.md` | 실제 clone 주소, Windows 설치·실행·사전점검 안내 추가 |
| `requirements.txt` | Windows TensorFlow 조건 수정, OpenCV contrib 사용 |
| `requirements-windows.txt` | 검증된 Windows 직접 의존성 버전 고정 |
| `setup_windows.ps1` | 호환 Python 탐색, 가상환경 및 패키지 자동 설치 |
| `run_windows.bat` | UTF-8 환경에서 GUI를 콘솔 없이 실행 |
| `preflight_windows.py` | 모델·카메라·MediaPipe 발표 전 자동점검 |
| `utils.py` | DirectShow 우선 공통 `open_camera()` 추가 |
| `model.py` | `tf.function` 기반 실시간 추론 함수 추가 |
| `gui.py` | 공통 카메라 사용, 초기 미리보기, 최적화 추론, 로딩 화면 |
| `translator.py` | 공통 카메라와 최적화 추론 적용 |
| `gesture_actions.py` | 공통 카메라와 최적화 추론 적용 |
| `data_collector.py` | 공통 카메라 적용 |
| `debug.py` | 공통 카메라 적용 |

전체 변경량:

```text
14 files changed
305 insertions
29 deletions
```

---

## 12. 수행한 검증

### 정적 검증

- 전체 Python 파일 `compileall` 통과
- `git diff --check` 통과
- PowerShell 스크립트 구문 분석 통과
- `pip check` 결과: 손상된 의존성 없음
- Windows에서 `requirements.txt`가 `tensorflow-cpu`를 정상 선택하는 것 확인

### 모델 검증

- `sign_model.keras` 로딩 성공
- 입력 형태 `(None, 30, 249)` 확인
- 출력 형태 `(None, 2)` 확인
- 파라미터 수 `248,002` 확인
- 레이블 수와 출력 클래스 수 일치 확인
- 더미 입력 확률 합 `1.0` 확인
- 기존 및 최적화 추론 결과 완전 일치 확인

### 하드웨어 검증

- Windows 웹캠 열기 성공
- 1280×720 프레임 읽기 성공
- DirectShow 백그라운드 스레드 읽기 성공
- MediaPipe Tasks 손 추적 모델 생성 성공
- 랜드마크 벡터 `(249,)` 생성 성공

### GUI 검증

- GUI 생성 성공
- 모델 상태 `MODEL OK | 2 labels` 확인
- 첫 프레임 약 2.16초 확인
- 20초 연속 GUI 업데이트 성공
- 약 19 FPS 표시 확인
- 정상 종료 확인

---

## 13. 아직 해결되지 않은 제한사항

### 13.1 실제 인식 정확도 미검증

저장소에는 현재 학습 데이터가 포함되어 있지 않고, 모델 레이블도 `만나다`, `안녕` 두 개뿐이다. 학습에 사용되지 않은 사람과 촬영 환경으로 구성한 독립 테스트 세트가 없어 다음 항목을 계산할 수 없다.

- 전체 정확도
- 레이블별 정밀도와 재현율
- 사람별 일반화 성능
- 조명 및 배경 변화에 대한 견고성
- 비수어 동작 거부율

학습 그래프에서 검증 정확도가 100%에 도달하더라도, 데이터 분할 방식과 표본 수를 확인할 수 없으므로 실제 발표 정확도의 증거로 사용할 수 없다.

### 13.2 거부 클래스 부족

Softmax 모델은 입력에 대해 항상 기존 레이블 중 하나를 선택한다. 현재 두 레이블만 있으면 임의의 손동작도 `만나다` 또는 `안녕`으로 분류될 수 있다.

발표 전 `모름` 또는 `기타` 클래스를 추가하고 다음 데이터를 포함하는 것이 필요하다.

- 손을 가만히 둔 상태
- 수어 사이의 전환 동작
- 얼굴이나 머리를 만지는 동작
- 임의의 손 흔들기
- 카메라에 손이 일부만 보이는 상태
- 좌우 손 위치가 바뀐 상태

### 13.3 실행 FPS와 인식 지연은 다름

GUI가 약 19 FPS로 표시되더라도 실제 단어 확정에는 최소 다음 과정이 필요하다.

1. 30개의 유효 프레임 수집
2. 동작량 기준 통과
3. 동일 예측이 `STABLE_FRAMES` 동안 유지
4. 모든 예측이 신뢰도 임계값 통과

따라서 화면이 부드럽게 보이는 것과 단어가 빠르게 확정되는 것은 별개의 문제다. 발표 동작 속도에 맞춰 `SEQUENCE_LENGTH`, `STABLE_FRAMES`, 임계값을 실제 사용자 데이터로 조정해야 한다.

### 13.4 Python은 별도 설치 필요

Windows 실행 스크립트는 Python 자체를 자동 설치하지 않는다. Python 3.10~3.12 64-bit가 없으면 안내 메시지를 출력하고 종료한다. 발표 PC에는 Python 3.12 64-bit 설치를 권장한다.

### 13.5 Windows 경로 길이

TensorFlow는 내부 헤더 경로가 매우 길다. 긴 임시 폴더 아래에 가상환경을 만들었을 때 Windows 경로 길이 제한으로 설치가 실패하는 현상이 실제로 발생했다.

프로젝트는 가능한 한 다음처럼 짧은 경로에 두는 것이 안전하다.

```text
C:\Hand-Bridge
```

---

## 14. 발표 전 권장 체크리스트

### 일주일 전

- [ ] 발표용 PC에 Python 3.12 64-bit 설치
- [ ] `setup_windows.ps1` 실행
- [ ] `preflight_windows.py` 전체 통과 확인
- [ ] 실제 발표할 사람으로 데이터 추가 수집
- [ ] `모름/기타` 클래스 추가
- [ ] 학습에 사용하지 않은 사람으로 인식률 측정

### 하루 전

- [ ] 발표 장소와 유사한 조명에서 테스트
- [ ] 실제 사용할 카메라와 USB 포트 고정
- [ ] Windows 카메라 앱, Zoom, Teams 등 카메라 점유 프로그램 종료
- [ ] 절전 및 화면 꺼짐 설정 확인
- [ ] `run_windows.bat` 더블클릭 실행 확인
- [ ] 발표할 수어 순서대로 10회 이상 반복 테스트
- [ ] 실패에 대비한 녹화 시연 영상 준비

### 발표 직전

- [ ] `preflight_windows.py` 실행
- [ ] 카메라 렌즈 및 배경 확인
- [ ] 손 전체가 프레임 안에 들어오는 거리 확인
- [ ] 모델 상태 `MODEL OK` 확인
- [ ] 카메라 FPS가 안정화될 때까지 대기
- [ ] 네트워크 없이도 실행되는지 확인

---

## 15. 결론

이번 개선으로 기존 Hand-Bridge의 Windows 설치 실패, GUI 카메라 멈춤, 단일 샘플 추론 오버헤드, 시작 상태 표시 부족을 수정했다. 검증 환경에서는 모델과 웹캠을 포함한 GUI가 정상 실행됐고 약 19 FPS로 표시됐으며, 최적화된 모델 추론은 기존 대비 큰 폭으로 개선됐다.

다만 현재 결과는 **Windows 실행성과 처리 성능을 검증한 것**이다. 저장된 모델의 실제 수어 인식 정확도와 새로운 사용자에 대한 일반화 성능은 아직 검증되지 않았다. 수료식 발표에서는 프로젝트를 “두 단어를 대상으로 한 실시간 한국수어 인식 프로토타입”으로 정확하게 설명하고, 거부 클래스와 독립 테스트 데이터를 추가한 뒤 정확도 수치를 제시하는 것이 바람직하다.
