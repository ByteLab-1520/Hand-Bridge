<div align="center">

```
██╗  ██╗ █████╗ ███╗   ██╗██████╗       ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
██║  ██║██╔══██╗████╗  ██║██╔══██╗      ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
███████║███████║██╔██╗ ██║██║  ██║█████╗██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗
██╔══██║██╔══██║██║╚██╗██║██║  ██║╚════╝██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝
██║  ██║██║  ██║██║ ╚████║██████╔╝      ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
```

**Real-time Korean Sign Language (KSL) translator — fully local, no cloud, no GPU required.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow CPU](https://img.shields.io/badge/TensorFlow-CPU--only-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-0097A7?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-00ff88?style=flat-square)](LICENSE)

</div>

---

## ░░ What is Hand-Bridge?

**Hand-Bridge** bridges the communication gap between Korean Sign Language (KSL / 한국수어) users and the hearing world.
It runs **entirely on your machine** — no internet, no cloud API, no GPU needed.

| | |
|---|---|
| 📷 | Webcam-based real-time translation |
| 🤖 | LSTM model trained on your own data |
| 🖐 | MediaPipe landmark tracking for hands, with optional face/body capture |
| 🇰🇷 | Korean Sign Language (KSL / 한국수어) |
| 🎮 | Pixel-art UI |
| 💻 | Windows · macOS · Linux |

---


## ░░ How It Works

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Webcam  │───▶│  MediaPipe  │───▶│  LSTM Model  │───▶│ Korean Text │
│  (live)  │    │ Landmarks   │    │  (30 frames) │    │   Output    │
└──────────┘    └─────────────┘    └──────────────┘    └─────────────┘
                  hands/face/body      249 features
                  zero-padded          per frame
```

1. **MediaPipe** detects hand landmarks, and can optionally include selected face or upper-body pose landmarks
2. Every frame is converted into a fixed **249-feature** vector; unused capture sections are zero-filled
3. A sliding window of **30 frames** is fed into an **LSTM** network
4. The model outputs a Korean word when a stable prediction crosses the confidence threshold
5. If no meaningful input is detected for **3 seconds**, the translation box is cleared automatically

---

## ░░ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/hand-bridge.git
cd hand-bridge

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add Training Videos

Organize your `.mp4` / `.mov` files by word:

```
videos/
├── 안녕하세요/
│   ├── annyeong_you_01.mp4
│   └── annyeong_friend_01.mp4
├── 감사합니다/
│   └── gamsahamnida_you_01.mp4
└── ...
```

> **Tip:** 30fps · 1080p · 2–3 seconds per clip works best.

### 3. Import Videos → Extract Landmarks

```bash
python video_importer.py --review   # preview pose estimation before saving
# or
python video_importer.py            # auto import
```

### 4. Train the Model

```bash
python train.py
```

Training saves:

```
models/sign_model.keras
models/labels.json
models/training_history.png
```

`train.py` skips malformed `.npy` sequences, checks that at least two labels have enough valid data, and prints a short actionable error instead of a long traceback when data is insufficient.

### 5. Run the GUI

```bash
python gui.py
```

---

## ░░ GUI Overview

```
░░ Hand-Bridge  //  Real-time Korean Sign Language Translator ░░
┌─────────────┬──────────────────────────────────────────────────┐
│  [ MENU ]  │  [ CAMERA FEED ]                      30 FPS      │
│            │  ┌────────────────────────────────────┐           │
│ ▶ 번역기     │  │  live feed + hand landmark overlay │           │
│   데이터     │  └────────────────────────────────────┘           │
│   가져오기    │                                                  │
│   학습       │  [ CURRENT SIGN ]   안녕하세요                      │
│             │  [ CONFIDENCE ]     ████████░░  87%              │
│ v1.0 CPU    │  [ TRANSLATION ]    안녕하세요 감사합니다              │
└─────────────┴──────────────────────────────────────────────────┘
```

| Tab | Description |
|-----|-------------|
| `▶ 번역기` | Live webcam translation |
| `◉ 데이터 수집` | Record signs directly via webcam |
| `↓ 영상 가져오기` | Convert MP4 files into training data |
| `⚙ 학습` | Train the LSTM model + view logs |

---

## ░░ Translator Behavior

The translator is tuned for short live demonstrations:

| Behavior | Current rule |
|----------|--------------|
| Input window | 30 frames |
| Stable prediction | Same prediction for `STABLE_FRAMES` frames |
| Confidence threshold | `PREDICTION_THRESHOLD` in `config.py` |
| Static-hand filtering | Ignores frames with too little hand motion/displacement |
| Auto clear | Clears `TRANSLATION` after 3 seconds with no hand or no meaningful motion |
| Capture mode | Current trained models use `두 손` mode by default |

Because the model is a softmax classifier, it will always choose one of the labels it was trained on. For better rejection of non-sign gestures, collect an extra label such as `모름` / `기타` with idle hands, transition poses, and random non-sign motion.

---

## ░░ Webcam Data Collection

The `◉ 데이터 수집` tab records training data directly from the webcam.

Current collection flow:

1. Enter a Korean label and the number of sequences.
2. Press `[ REC ] 녹화 시작`.
3. The app waits until a hand is detected.
4. Once a hand appears, a 3-second countdown starts.
5. The sequence records only frames where a hand is detected.
6. When 30 valid frames are collected, the sequence is saved.
7. The next sequence again waits for hand detection before counting down.

Controls while collecting:

| Control | Action |
|---------|--------|
| `[ ⏸ ] 일시정지` | Pause/resume the current sequence |
| `[ X ] 취소` | Discard the current sequence and retry it |
| Auto cancel | If hands disappear for about 3 seconds during recording, the current partial sequence is discarded |

The CLI collector has matching keyboard controls:

| Key | Action |
|-----|--------|
| `SPACE` | Start / pause / resume |
| `P` | Pause / resume |
| `C` | Cancel the current sequence and discard collected frames |
| `Q` | Quit |

---

## ░░ Utilities

### Delete Training Data & Models

Use `delete.py` to safely remove training data and/or models:

```bash
# Delete training data only (data/ folder)
python delete.py

# Delete both training data AND model
python delete.py --all

# Delete model only (keep training data)
python delete.py --model-only
```

Each command will ask for confirmation before deletion (`yes/no`).

---

## ░░ Project Structure

```
hand-bridge/
├── gui.py               ← Main app (pixel UI)
├── translator.py        ← CLI translator
├── data_collector.py    ← CLI webcam collector
├── video_importer.py    ← MP4 → training data converter
├── train.py             ← Model training script
├── delete.py            ← Data/model deletion utility
├── model.py             ← LSTM architecture
├── utils.py             ← Landmark extraction + Korean text rendering
├── config.py            ← All settings (camera, model, thresholds)
├── requirements.txt
├── data/                ← Collected landmark sequences (.npy)
├── models/              ← Trained model + labels
└── videos/              ← Your training videos (MP4/MOV)
```

---

## ░░ Recording Guidelines

| Setting | Recommended |
|---------|-------------|
| Resolution | 1920×1080 |
| Frame rate | 30 fps (60 fps also works) |
| Clip length | 2–3 seconds |
| Format | `.mp4` (H.264 / HEVC) or `.mov` |
| Background | Plain single-color wall |
| Lighting | Bright, even, no backlight |
| Distance | 50–80 cm from camera |
| Sequences per word | 30 minimum · 50–100 recommended |
| Rejection class | Add `모름` / `기타` with 80–100 idle or non-sign examples |

**File naming:** `<word>_<name>_<number>.mp4`
→ e.g. `annyeong_jiseong_01.mp4`

For live demos, start small: 5–8 signs plus one `모름` class is usually more reliable than adding many similar signs with too few examples.

---

## ░░ Collaborative Training

Hand-Bridge is designed for **group data collection**. The more people contribute, the better the model generalizes.

```
Friend A  →  videos/안녕하세요/annyeong_minho_01.mp4
Friend B  →  videos/안녕하세요/annyeong_sora_01.mp4
You       →  videos/안녕하세요/annyeong_you_01.mp4
                          ↓
               python video_importer.py
               python train.py
```

To merge datasets from multiple PCs, copy the `data/` folder and renumber any overlapping `.npy` files.

---

## ░░ Portability

| Scenario | What to copy | Retrain? |
|----------|-------------|----------|
| Run on a new PC | `models/` + all `.py` files | ❌ No |
| Add new signs | Full project folder | ✅ Yes |
| Share model with friends | `models/sign_model.keras` + `models/labels.json` | ❌ No |

The model is **cross-platform** (Windows ↔ macOS ↔ Linux) thanks to `tensorflow-cpu`.

---

## ░░ Requirements

```
Python       3.10+
mediapipe    0.10+
tensorflow   CPU-only build (no CUDA / Metal required)
opencv       4.8+
customtkinter 5.2+
Pillow       10+
```

---

## ░░ License

MIT © 2025 — See [LICENSE](LICENSE)

---

<div align="center">

**Hand-Bridge** — _Connecting hands to words._

</div>
