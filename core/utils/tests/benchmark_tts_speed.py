import torch
import time
import numpy as np
from bot.core.tts.tts_client import TTSClient
from bot.core.utils.config import Config

def benchmark_tts():
    print(f"Loading TTS model on {Config.DEVICE}...")
    start_load = time.time()
    tts = TTSClient()
    print(f"Model loaded in {time.time() - start_load:.2f}s")
    
    # Check devices
    model = tts.model.model
    print(f"Main model device: {model.device}, dtype: {model.dtype}")
    print(f"Talker device: {model.talker.device}")
    print(f"Code predictor device: {next(model.talker.code_predictor.parameters()).device}")
    print(f"Speaker encoder device: {next(model.speaker_encoder.parameters()).device if model.speaker_encoder else 'N/A'}")

    text = "我是MyBot，你的智能语音助手，随时为你提供帮助！"
    print(f"Generating audio for text: '{text}'")
    
    # First call (includes compilation/warmup overhead if any)
    start_gen = time.time()
    audio, sr = tts.generate(text)
    print(f"First generation took {time.time() - start_gen:.2f}s")
    
    # Second call
    start_gen = time.time()
    audio, sr = tts.generate(text)
    print(f"Second generation took {time.time() - start_gen:.2f}s")

    # Short sentence
    text_short = "你好。"
    start_gen = time.time()
    audio, sr = tts.generate(text_short)
    print(f"Short generation ('{text_short}') took {time.time() - start_gen:.2f}s")

if __name__ == "__main__":
    benchmark_tts()
