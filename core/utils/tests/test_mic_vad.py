import sys
import os
import wave
import logging
import pyaudio

# Add project root to sys.path to allow imports from bot.*
# Assuming this script is at bot/core/utils/tests/test_mic_vad.py
# We need to go up 4 levels to reach the root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bot.core.asr.microphone import Microphone
from bot.core.asr.vad import VadDetector
from bot.core.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_wav(data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(Config.CHANNELS)
        wf.setsampwidth(2) # 16-bit PCM
        wf.setframerate(Config.SAMPLE_RATE)
        wf.writeframes(data)
    logger.info(f"Saved audio to {filename}")

def main():
    detector = VadDetector()
    
    logger.info("Initializing Microphone...")
    logger.info(f"VAD Parameters: Mode={Config.VAD_MODE}, Frame={Config.VAD_FRAME_MS}ms")
    logger.info("Please speak into the microphone. Recording will stop automatically after one sentence.")
    
    output_file = os.path.join(current_dir, "test_vad_output.wav")

    try:
        with Microphone() as mic:
            for chunk in mic.stream():
                speech_segment = detector.process_chunk(chunk)
                
                if speech_segment:
                    logger.info(f"Captured speech segment of {len(speech_segment)} bytes.")
                    save_wav(speech_segment, output_file)
                    break
        
        logger.info("Test passed successfully.")
                    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")

if __name__ == "__main__":
    main()
