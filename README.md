# Reachy Mini Spark Voice Assistant

Full-duplex voice assistant for the **Reachy Mini Wireless** robot.  
Speech processing (STT + TTS + VAD) runs locally on the robot's **CM4**, while all LLM inference is offloaded to the **Spark** via its OpenAI-compatible HTTP API.  Home Assistant integration is ready to wire in whenever needed.

---

## Architecture

```
┌──────────────────────────── Reachy Mini (CM4) ──────────────────────────────┐
│                                                                               │
│  ReSpeaker mic array                                                          │
│        │                                                                      │
│        ▼                                                                      │
│  ┌───────────┐    float32   ┌──────────┐   text   ┌──────────────────────┐  │
│  │  VAD      │ ──────────▶  │ Whisper  │ ───────▶  │  Spark LLM client   │  │
│  │(webrtcvad)│  (16 kHz)    │(tiny/sm) │           │  (OpenAI SDK)       │  │
│  └───────────┘              └──────────┘           └──────────┬───────────┘  │
│                                                               │ reply text   │
│  Built-in speaker                                             ▼              │
│        ▲                              ┌──────────┐   audio   │              │
│        └──────────────────────────── │  Piper   │ ◀─────────┘              │
│                                       │  TTS     │                           │
│                                       └──────────┘                           │
│                                                                               │
│  Head / antenna behaviours ◀──── robot.py (non-blocking)                    │
└───────────────────────────────────────────────────────────────────────────────┘
          │ OpenAI-compat HTTP                          │ REST API (optional)
          ▼                                              ▼
    ┌───────────┐                               ┌───────────────┐
    │  Spark    │                               │ Home Assistant│
    │  (LLM)   │                               │   (HASS)      │
    └───────────┘                               └───────────────┘
```

### Component responsibilities

| Component | Where it runs | Library |
|---|---|---|
| Voice Activity Detection | CM4 | `webrtcvad` |
| Speech-to-Text | CM4 | `faster-whisper` (tiny/small model) |
| Text-to-Speech | CM4 | `piper-tts` + espeak fallback |
| LLM | **Spark** | OpenAI-compatible API (Ollama, LM Studio, …) |
| Home automation | Home Assistant | REST API |

---

## Quick start

### 1. Install

```bash
chmod +x install.sh
./install.sh
```

The installer handles:
- System packages (espeak-ng, GStreamer, etc.)
- Python virtualenv at `~/.venvs/spark-voice`
- Piper binary for your architecture (arm64/armv7l/amd64)
- Optional systemd service

### 2. Configure

Edit **`config.yaml`** (all secrets can alternatively be set as env-vars):

```yaml
spark:
  api_base: "http://spark.local:11434/v1"   # ← point to your Spark
  model: "llama3.2:3b"                        # ← model loaded on Spark

hass:
  enabled: true                               # ← flip to true for HASS
  url: "http://homeassistant.local:8123"
  token: "ey..."                              # ← HA long-lived token
  exposed_entities:
    - light.living_room
    - switch.fan
```

Environment-variable overrides (useful for secrets):

| Variable | Config key |
|---|---|
| `SPARK_API_BASE` | `spark.api_base` |
| `SPARK_API_KEY` | `spark.api_key` |
| `SPARK_MODEL` | `spark.model` |
| `HASS_URL` | `hass.url` |
| `HASS_TOKEN` | `hass.token` |

### 3. Run

```bash
# Manual (development)
source ~/.venvs/spark-voice/bin/activate
python -m spark_voice.conversation --log-level DEBUG

# Via systemd
sudo systemctl enable --now spark-voice
journalctl -fu spark-voice
```

---

## Conversation flow

1. **Listen** – VAD waits for speech, accumulates audio until 1.2 s of silence  
2. **Transcribe** – Whisper converts speech to text on-device  
3. **LLM** – text + optional HASS state context sent to Spark; response received  
4. **Action** – any `[HASS:…]` tag extracted and dispatched to Home Assistant  
5. **Speak** – Piper synthesises the reply; played through the built-in speaker  
6. **Behave** – head nods, antenna wiggles, and thinking tilts run in parallel  

---

## Home Assistant integration

When `hass.enabled: true`, the assistant:

- **Injects device states** into every LLM prompt so the model knows what's on/off  
- **Parses action tags** from the LLM response:
  ```
  [HASS:light/turn_on:light.living_room:{"brightness":200}]
  ```
- **Executes the service call** before speaking the reply

The LLM prompt instructs the model to emit at most one action tag per turn.  
Add more entities to `hass.exposed_entities` to give the model more control.

---

## STT model size trade-offs (CM4)

| Model | Accuracy | ~Latency (CM4) | Disk |
|---|---|---|---|
| `tiny` | good | ~1 s | 75 MB |
| `base` | better | ~2 s | 145 MB |
| `small` | best practical | ~5 s | 466 MB |

Set `stt.model` in `config.yaml`.

---

## TTS options

| Engine | Quality | Setup |
|---|---|---|
| Piper (`en_US-lessac-medium`) | Natural neural voice | Auto-downloaded |
| Piper (`en_US-ryan-high`) | Highest quality | Auto-downloaded |
| espeak-ng | Robotic fallback | Always available |

Set `tts.engine: piper` or `tts.engine: espeak` in `config.yaml`.

---

## Adding wake-word support

Install `openwakeword` and set `wake_word.enabled: true` in `config.yaml`.  
The conversation loop can then be extended in `conversation.py` to call the
openwakeword detector before starting VAD collection.  The scaffolding is
already wired in the config; code hook is marked `# TODO: wake-word` in
`conversation.py`.

---

## File layout

```
reachy-mini-spark/
├── spark_voice/
│   ├── __init__.py
│   ├── config.py          # YAML + env-var loader
│   ├── vad.py             # webrtcvad utterance collector
│   ├── stt.py             # faster-whisper STT
│   ├── tts.py             # Piper / espeak TTS
│   ├── llm.py             # Spark LLM client + HASS tag parser
│   ├── hass.py            # Home Assistant REST client
│   ├── robot.py           # Head/antenna behaviours
│   └── conversation.py    # ReachyMiniApp entry point
├── config.yaml            # All runtime settings
├── requirements.txt
├── setup.py
├── install.sh
└── README.md
```
