import sys
import os
import signal
import logging
import time
import numpy as np

# Add workspace root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from bot.core.asr.microphone import Microphone
from bot.core.asr.vad import VadDetector
from bot.core.asr.asr_client import ASRClient
from bot.core.llm.llm_client import OpenAILLMClient
from bot.core.tts.tts_client import TTSClient
from bot.core.tts.player import AudioPlayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MyBot")

class MyBot:
    def __init__(self):
        self.running = True
        self.history = []
        self.system_prompt = "你是一个智能语音助手，你的名字叫MyBot。请用简明扼要、口语化的语言回答用户的问题。不要使用Markdown格式。"
        self.last_playing_time = 0 # Track when the bot last spoke
        
        try:
            self.asr = ASRClient()
            self.llm = OpenAILLMClient()
            self.tts = TTSClient()
            self.player = AudioPlayer()
            self.vad = VadDetector()
            self.mic = Microphone()
            
            self._warmup()
        except Exception as e:
            logger.critical(f"Initialization failed: {e}")
            raise

    def _warmup(self):
        """Warm up models to avoid first-time latency."""
        logger.info("Warming up models...")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.asr.transcribe(dummy_audio)
        self.tts.generate("你好", language="Chinese")
        logger.info("Warmup complete.")

    def run(self):
        self.player.start()
        logger.info("MyBot (Turn-based) Ready.")
        
        def signal_handler(sig, frame):
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)

        try:
            with self.mic as microphone:
                for chunk in microphone.stream():
                    if not self.running: break
                    
                    # --- Advanced Echo & Noise Suppression ---
                    # 1. If currently playing
                    if self.player.is_playing():
                        self.last_playing_time = time.time()
                        self.vad.reset()
                        continue
                    
                    # 2. Post-playback cooldown (800ms)
                    if time.time() - self.last_playing_time < 0.8:
                        self.vad.reset()
                        continue
                    
                    utterance = self.vad.process_chunk(chunk)
                    if utterance:
                        start_time = time.time()
                        
                        # 1. ASR
                        logger.info("Processing ASR...")
                        text, lang = self.asr.transcribe(utterance)
                        
                        # Filter out empty or too short junk inputs (echo/noise)
                        if not text or len(text.strip()) <= 1:
                            logger.info(f"Ignored short/empty input: '{text}'")
                            self.vad.reset()
                            continue
                            
                        logger.info(f"User: {text}")
                        self.history.append({"role": "user", "content": text})
                        
                        # 2. LLM
                        logger.info("Thinking (LLM)...")
                        full_response = ""
                        for token in self.llm.stream_chat(self.history, system_prompt=self.system_prompt):
                            if not self.running: break
                            full_response += token
                            if "</think>" in full_response:
                                full_response = full_response.split("</think>")[-1]
                        
                        full_response = full_response.strip()
                        if not full_response:
                            self.vad.reset()
                            continue
                            
                        self.history.append({"role": "assistant", "content": full_response})
                        logger.info(f"AI: {full_response}")
                        
                        # 3. TTS
                        logger.info("Synthesizing (TTS)...")
                        tts_start = time.time()
                        audio_data, sr = self.tts.generate(full_response, language=lang if lang else "Chinese")
                        tts_duration = time.time() - tts_start
                        
                        if audio_data is not None:
                            logger.info(f"Playback starting (TTS took {tts_duration:.2f}s)...")
                            self.player.play(audio_data)
                            # Update last_playing_time to now so cooldown starts properly
                            self.last_playing_time = time.time()
                        
                        total_duration = time.time() - start_time
                        logger.info(f"Turn completed in {total_duration:.2f}s.")
                        self.vad.reset()

        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.player.stop()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    bot = MyBot()
    bot.run()
