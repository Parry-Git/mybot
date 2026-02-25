import pyaudio
import os
import torch

class Config:
    # Audio Hardware Settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    
    # VAD Settings (WebRTC VAD)
    # Mode 0-3 (3 is most aggressive in filtering out non-speech)
    VAD_MODE = 3
    # Frame duration in ms (10, 20, or 30ms supported by webrtcvad)
    VAD_FRAME_MS = 30
    # Number of samples per frame
    # 16000Hz * 30ms / 1000 = 480 samples
    VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
    
    # VAD Logic Settings
    # How many frames of silence to assume end of speech
    # 30ms * 20 = 600ms silence
    SILENCE_LIMIT_FRAMES = 20
    # How many frames of speech to trigger start of recording
    # 30ms * 5 = 150ms speech
    SPEECH_TRIGGER_FRAMES = 5
    
    # Paths (Placeholder for future model paths)
    ASR_MODEL_PATH = "Qwen/Qwen3-ASR-0.6B"
    TTS_MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    # System Settings
    DEVICE = "cuda" # "cuda", "cpu", "cuda:0"
    DTYPE = torch.bfloat16 # Use bfloat16 for RTX 30/40 series
    # Attention implementation: "eager", "sdpa", "flash_attention_2"
    ATTN_IMPLEMENTATION = "flash_attention_2"
    
    # TTS Configuration
    TTS_SAMPLE_RATE = 24000
    TTS_REF_AUDIO_PATH = os.path.join("bot", "assets", "caixukun2.wav")
    # TTS_REF_AUDIO_PATH = os.path.join("assets", "caixukun2.wav")

    DEEPSEEK_API_KEY = "REMOVED_API_KEY"

