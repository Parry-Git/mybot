import sys
import logging
import os
import signal
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bot.core.asr.microphone import Microphone
from bot.core.asr.vad import VadDetector
from bot.core.asr.asr_client import ASRClient
from bot.core.tts.tts_client import TTSClient
from bot.core.tts.player import AudioPlayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EchoBot")

def main():
    logger.info("Initializing Echo Bot...")
    
    # Initialize components
    try:
        asr = ASRClient()
        tts = TTSClient()
        player = AudioPlayer()
        vad = VadDetector()
        mic = Microphone()
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    player.start()
    
    logger.info("Echo Bot Ready. Speak into the microphone.")
    
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Interrupt received, stopping...")
        running = False
        
    signal.signal(signal.SIGINT, signal_handler)

    try:
        with mic as microphone:
            for chunk in microphone.stream():
                if not running:
                    break
                    
                utterance = vad.process_chunk(chunk)
                
                if utterance:
                    logger.info(f"Captured utterance ({len(utterance)} bytes). Transcribing...")
                    
                    # ASR
                    text, lang = asr.transcribe(utterance)
                    if not text:
                        logger.info("ASR could not recognize speech or empty.")
                        continue
                    
                    logger.info(f"User said: {text} (Language: {lang})")
                    
                    # TTS
                    logger.info("Synthesizing response...")
                    # Normalize language name if necessary, TTSClient handles generic names usually
                    # Qwen3-ASR returns "Chinese", "English" etc.
                    
                    audio_data, sr = tts.generate(text, language=lang if lang else "Chinese")
                    
                    if audio_data is not None:
                        logger.info(f"Playing response ({len(audio_data)} samples at {sr}Hz)...")
                        player.play(audio_data)
                    else:
                        logger.warning("TTS failed to generate audio.")

    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        player.stop()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
