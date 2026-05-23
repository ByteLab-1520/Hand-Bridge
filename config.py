import os

# Force CPU-only execution — no CUDA, no Metal, no ROCm.
# Set these before TensorFlow is imported anywhere in the process.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'       # hide all CUDA GPUs
os.environ['TF_METAL_DEVICE_ENABLE'] = '0'      # disable Apple Metal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # suppress device-probe noise

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS_DIR  = MODEL_DIR  # alias for task model files
MODEL_PATH  = os.path.join(MODEL_DIR, 'sign_model.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')

# Data collection settings
SEQUENCE_LENGTH = 30        # number of frames per sign sequence
NUM_SEQUENCES = 30          # sequences to collect per label (default)
# 21 landmarks * 3 coords (x,y,z) * 2 hands = 126 features
HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
MAX_HANDS = 2
# MediaPipe confidence thresholds
DETECTION_CONFIDENCE = 0.65
TRACKING_CONFIDENCE = 0.7

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Key face landmark indices for expression recognition (24 points)
# Covers mouth, eyebrows, eyes, nose, and cheeks
KEY_FACE_INDICES = [
    # Mouth
    61, 291, 0, 17, 13, 14, 78, 308,
    # Eyebrows
    70, 300, 107, 336,
    # Eyes
    33, 263, 133, 362, 159, 386, 145, 374,
    # Nose
    4, 168,
    # Cheeks
    234, 454,
]
NUM_KEY_FACE = len(KEY_FACE_INDICES)  # 24

# Upper body pose landmark indices (MediaPipe Pose 0–16)
POSE_INDICES = list(range(17))
NUM_KEY_POSE = len(POSE_INDICES)  # 17

# Total features per frame:
#   hands 126  +  key face 72  +  upper body pose 51  =  249
NUM_FEATURES = (
    HAND_LANDMARKS * COORDS_PER_LANDMARK * MAX_HANDS  # 126
    + NUM_KEY_FACE * COORDS_PER_LANDMARK              # 72
    + NUM_KEY_POSE * COORDS_PER_LANDMARK              # 51
)  # = 249

# Model architecture
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.20  # 작은 데이터셋에서 confidence가 눌리지 않도록 낮게 유지
L2_REGULARIZATION = 0.00001  # 과한 정규화 방지

# Training settings
EPOCHS = 150  # 최대 에포크
BATCH_SIZE = 16  # 배치 크기 감소 (32 → 16) - 더 정교한 그래디언트
VALIDATION_SPLIT = 0.20  # 검증 데이터 비율 증가 (15% → 20%)
EARLY_STOPPING_PATIENCE = 15  # 조기 종료 patience 감소 (20 → 15) - 과적합 방지

# Inference settings
PREDICTION_THRESHOLD = 0.70   # minimum confidence (0.80 → 0.70, 균형잡힌 값)
STABLE_FRAMES = 10            # frames to stabilize (15 → 10, 더 빠른 인식)
MIN_GESTURE_MOTION = 0.0020   # 손이 가만히 있을 때 오인식 방지용 평균 프레임 변화량
MIN_GESTURE_DISPLACEMENT = 0.025  # 시작-끝 손 좌표 변화량
INACTIVITY_CLEAR_SECONDS = 3.0  # 번역 결과 자동 초기화 대기 시간

# Capture modes — determines which body parts are extracted as features
CAPTURE_TWO_HANDS = "두 손"        # both hands only (face/body zeroed)
CAPTURE_WITH_FACE = "얼굴 포함"    # both hands + facial landmarks
CAPTURE_WITH_BODY = "몸 포함"      # both hands + upper-body pose
CAPTURE_ONE_HAND  = "한 손"        # right hand only (left/face/body zeroed)

CAPTURE_MODES = [CAPTURE_TWO_HANDS, CAPTURE_WITH_FACE, CAPTURE_WITH_BODY, CAPTURE_ONE_HAND]
DEFAULT_CAPTURE_MODE = CAPTURE_TWO_HANDS
