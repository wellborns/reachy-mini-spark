from setuptools import setup, find_packages

setup(
    name="spark-voice",
    version="0.1.0",
    description="STT/TTS voice assistant for Reachy Mini Wireless using Spark LLM",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "reachy-mini",
        "faster-whisper>=1.0.0",
        "piper-tts>=1.2.0",
        "webrtcvad-wheels>=2.0.10",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "openai>=1.30.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "spark-voice=spark_voice.conversation:main",
        ]
    },
)
