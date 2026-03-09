import sys
import os
import signal
import logging
import time
import numpy as np
import argparse

# Add workspace root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from bot.core.llm.llm_client import OpenAILLMClient
from bot.core.tts.kokoro_tts_client import KokoroTTSClient
from bot.core.tts.player import AudioPlayer

# Conditionally import voice components
try:
    from bot.core.asr.microphone import Microphone
    from bot.core.asr.vad import VadDetector
    from bot.core.asr.asr_client import ASRClient
    VOICE_COMPONENTS_AVAILABLE = True
except ImportError:
    VOICE_COMPONENTS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MyBot")

class MyBot:
    def __init__(self, mode='voice'):
        self.running = True
        self.history = []
        self.system_prompt = "You are a smart voice assistant named MyBot. Please answer the user's questions in a concise and conversational manner. Do not use Markdown format."
        self.mode = mode
        
        try:
            self.llm = OpenAILLMClient()
            self.tts = KokoroTTSClient()
            self.player = AudioPlayer()

            if self.mode == 'voice':
                if not VOICE_COMPONENTS_AVAILABLE:
                    raise ImportError("Voice components are not available. Please install them to use voice mode.")
                self.asr = ASRClient()
                self.vad = VadDetector()
                self.mic = Microphone()
                self.last_playing_time = 0
            
            self._warmup()
        except Exception as e:
            logger.critical(f"Initialization failed: {e}")
            raise

    def _warmup(self):
        """Warm up models to avoid first-time latency."""
        logger.info("Warming up models...")
        self.tts.generate("Hello.")
        if self.mode == 'voice':
            dummy_audio = np.zeros(16000, dtype=np.float32)
            self.asr.transcribe(dummy_audio)
        logger.info("Warmup complete.")

    def run_voice_based(self):
        self.player.start()
        logger.info("MyBot (Voice-based) Ready.")
        
        def signal_handler(sig, frame):
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)

        try:
            with self.mic as microphone:
                for chunk in microphone.stream():
                    if not self.running: break
                    
                    if self.player.is_playing():
                        self.last_playing_time = time.time()
                        self.vad.reset()
                        continue
                    
                    if time.time() - self.last_playing_time < 0.8:
                        self.vad.reset()
                        continue
                    
                    utterance = self.vad.process_chunk(chunk)
                    if utterance:
                        start_time = time.time()
                        
                        logger.info("Processing ASR...")
                        text, lang = self.asr.transcribe(utterance)
                        
                        if not text or len(text.strip()) <= 1:
                            logger.info(f"Ignored short/empty input: '{text}'")
                            self.vad.reset()
                            continue
                            
                        logger.info(f"User: {text}")
                        self.history.append({"role": "user", "content": text})
                        
                        self._process_and_speak()
                        
                        total_duration = time.time() - start_time
                        logger.info(f"Turn completed in {total_duration:.2f}s.")
                        self.vad.reset()

        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.stop()

    def run_text_based(self):
        self.player.start()
        logger.info("MyBot (Text-based) Ready. Type 'quit' to exit.")
        
        def signal_handler(sig, frame):
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.running:
                text = input("You: ")
                if text.lower() == 'quit':
                    self.running = False
                    continue

                if not text or len(text.strip()) <= 1:
                    continue
                    
                self.history.append({"role": "user", "content": text})
                self._process_and_speak()

        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.stop()

    def _process_and_speak(self):
        logger.info("Thinking (LLM)...")
        full_response = ""
        for token in self.llm.stream_chat(self.history, system_prompt=self.system_prompt):
            if not self.running: break
            full_response += token
            if "</think>" in full_response:
                full_response = full_response.split("</think>")[-1]
        
        full_response = full_response.strip()
        if not full_response:
            return
            
        self.history.append({"role": "assistant", "content": full_response})
        logger.info(f"AI: {full_response}")
        
        logger.info("Synthesizing (TTS)...")
        tts_start = time.time()
        audio_data = self.tts.generate(full_response)
        sr = self.tts.get_sample_rate()
        tts_duration = time.time() - tts_start
        
        if audio_data is not None and audio_data.size > 0:
            logger.info(f"Playback starting (TTS took {tts_duration:.2f}s)...")
            self.player.play(audio_data, sample_rate=sr)
            if self.mode == 'voice':
                self.last_playing_time = time.time()

    def stop(self):
        self.running = False
        if self.player:
            self.player.stop()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyBot - A voice and text-based assistant.")
    parser.add_argument('--mode', type=str, default='voice', choices=['voice', 'text'],
                        help="Mode of operation: 'voice' for voice input, 'text' for text input.")
    args = parser.parse_args()

    bot = MyBot(mode=args.mode)
    if args.mode == 'voice':
        bot.run_voice_based()
    else:
        bot.run_text_based()
