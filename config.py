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
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Key face landmark indices for expression recognition (22 points)
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
NUM_KEY_FACE = len(KEY_FACE_INDICES)  # 22

# Upper body pose landmark indices (MediaPipe Pose 0–16)
POSE_INDICES = list(range(17))
NUM_KEY_POSE = len(POSE_INDICES)  # 17

# Total features per frame:
#   hands 126  +  key face 66  +  upper body pose 51  =  243
NUM_FEATURES = (
    HAND_LANDMARKS * COORDS_PER_LANDMARK * MAX_HANDS  # 126
    + NUM_KEY_FACE * COORDS_PER_LANDMARK              # 66
    + NUM_KEY_POSE * COORDS_PER_LANDMARK              # 51
)  # = 243

# Model architecture
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.4

# Training settings
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 20

# Inference settings
PREDICTION_THRESHOLD = 0.55   # minimum confidence to display prediction
STABLE_FRAMES = 10            # frames prediction must be stable before showing

# Capture modes — determines which body parts are extracted as features
CAPTURE_TWO_HANDS = "두 손"        # both hands only (face/body zeroed)
CAPTURE_WITH_FACE = "얼굴 포함"    # both hands + facial landmarks
CAPTURE_WITH_BODY = "몸 포함"      # both hands + upper-body pose
CAPTURE_ONE_HAND  = "한 손"        # right hand only (left/face/body zeroed)

CAPTURE_MODES = [CAPTURE_TWO_HANDS, CAPTURE_WITH_FACE, CAPTURE_WITH_BODY, CAPTURE_ONE_HAND]
DEFAULT_CAPTURE_MODE = CAPTURE_TWO_HANDS
