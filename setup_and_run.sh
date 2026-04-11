#!/usr/bin/env bash
# One-time setup + quick-launch helper
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Hand-Bridge 설정 ==="

# ── Python 3.12 확인 ──────────────────────────────────────────────────────────
# TensorFlow는 Python 3.9~3.12만 지원합니다.
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3.9; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "[오류] Python 3.9~3.12 를 찾을 수 없습니다."
    echo "현재 설치된 Python: $(python3 --version 2>&1)"
    echo ""
    echo "아래 명령으로 Python 3.12를 설치하세요:"
    echo "  brew install python@3.12"
    echo ""
    echo "설치 후 이 스크립트를 다시 실행하세요."
    exit 1
fi

echo "Python 버전: $($PYTHON --version)"

# ── 가상환경 생성 ─────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "가상환경 생성 중 ($PYTHON)..."
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate

echo "패키지 설치 중..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✓ 설치 완료!"
echo ""
echo "다음 명령으로 시작하세요:"
echo "  source .venv/bin/activate"
echo ""
echo "  1. GUI 실행:       python gui.py"
echo "  2. 영상 변환:      python video_importer.py --review"
echo "  3. 모델 학습:      python train.py"
