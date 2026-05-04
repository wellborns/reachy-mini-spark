#!/usr/bin/env bash
# install.sh – bootstrap the Spark Voice Assistant on Reachy Mini (CM4 / ARM64)
set -euo pipefail

echo "=== Reachy Mini Spark Voice Assistant – installer ==="

# -----------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------
echo "[1/5] Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-venv \
    espeak-ng \
    libespeak-ng-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-python3-plugin-loader \
    portaudio19-dev \
    wget

# -----------------------------------------------------------------------
# Python virtual environment
# -----------------------------------------------------------------------
echo "[2/5] Creating Python virtual environment …"
VENV_DIR="${HOME}/.venvs/spark-voice"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel

# -----------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------
echo "[3/5] Installing Python packages …"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"
pip install -e "${SCRIPT_DIR}"

# -----------------------------------------------------------------------
# Piper binary (if not already present)
# -----------------------------------------------------------------------
echo "[4/5] Checking piper TTS binary …"
PIPER_BIN="/usr/local/bin/piper"
if ! command -v piper &>/dev/null; then
    ARCH=$(uname -m)
    case "${ARCH}" in
        aarch64|arm64) PIPER_ARCH="arm64" ;;
        armv7l)        PIPER_ARCH="armv7l" ;;
        x86_64)        PIPER_ARCH="amd64" ;;
        *)             echo "Unknown arch ${ARCH}; skipping piper binary." ; PIPER_ARCH="" ;;
    esac

    if [ -n "${PIPER_ARCH}" ]; then
        PIPER_VERSION="2023.11.14-2"
        PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz"
        TMP=$(mktemp -d)
        wget -qO "${TMP}/piper.tar.gz" "${PIPER_URL}"
        tar -xzf "${TMP}/piper.tar.gz" -C "${TMP}"
        sudo install -m 755 "${TMP}/piper/piper" "${PIPER_BIN}"
        sudo cp "${TMP}/piper"/*.so* /usr/local/lib/ 2>/dev/null || true
        sudo ldconfig
        rm -rf "${TMP}"
        echo "  piper installed at ${PIPER_BIN}"
    fi
else
    echo "  piper already installed: $(command -v piper)"
fi

# -----------------------------------------------------------------------
# systemd service (optional)
# -----------------------------------------------------------------------
echo "[5/5] Installing systemd service …"
SERVICE_FILE="/etc/systemd/system/spark-voice.service"
sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=Reachy Mini Spark Voice Assistant
After=network.target reachy-mini.service
Wants=reachy-mini.service

[Service]
Type=simple
User=reachy
WorkingDirectory=${SCRIPT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${VENV_DIR}/bin/python -m spark_voice.conversation
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo ""
echo "=== Installation complete ==="
echo ""
echo "Before starting, edit config.yaml and set:"
echo "  spark.api_base  – URL of the Spark's LLM API (e.g. http://spark.local:11434/v1)"
echo "  hass.url        – Home Assistant URL (optional)"
echo "  hass.token      – HA long-lived access token (optional)"
echo ""
echo "To run manually:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python -m spark_voice.conversation --log-level DEBUG"
echo ""
echo "To enable as a system service:"
echo "  sudo systemctl enable --now spark-voice"
