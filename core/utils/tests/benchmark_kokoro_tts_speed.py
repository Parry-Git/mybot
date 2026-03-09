import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
import torch
import time
import numpy as np
from bot.core.tts.kokoro_tts_client import KokoroTTSClient
from bot.core.utils.config import Config

def benchmark_kokoro_tts():
    print(f"Loading Kokoro TTS model on {Config.DEVICE}...")
    start_load = time.time()
    tts = KokoroTTSClient()
    print(f"Model loaded in {time.time() - start_load:.2f}s")
    
    text = "I am MyBot, your intelligent voice assistant, ready to help you at any time!"
    print(f"Generating audio for text: '{text}'")
    
    # First call (includes compilation/warmup overhead if any)
    start_gen = time.time()
    audio = tts.generate(text)
    sr = tts.get_sample_rate()
    print(f"First generation took {time.time() - start_gen:.2f}s")
    
    # Second call
    start_gen = time.time()
    audio = tts.generate(text)
    print(f"Second generation took {time.time() - start_gen:.2f}s")

    # Short sentence
    text_short = "Hello."
    start_gen = time.time()
    audio = tts.generate(text_short)
    print(f"Short generation ('{text_short}') took {time.time() - start_gen:.2f}s")

if __name__ == "__main__":
    benchmark_kokoro_tts()
