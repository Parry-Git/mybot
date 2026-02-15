import sys
import os
import logging
import soundfile as sf

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bot.core.asr.asr_client import ASRClient
from bot.core.tts.tts_client import TTSClient
from bot.core.utils.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase2Verify")

def main():
    print("-" * 50)
    logger.info("Starting Phase 2 Logic Verification")
    print("-" * 50)

    # 1. Check Resources
    ref_audio_path = Config.TTS_REF_AUDIO_PATH
    if not os.path.exists(ref_audio_path):
        logger.error(f"CRITICAL: Reference audio not found at {ref_audio_path}")
        return
    logger.info(f"Resource Check: Reference audio found at {ref_audio_path}")

    # 2. Initialize Clients (Loads Models)
    try:
        logger.info("Initializing ASR Client (Loading Qwen3-ASR)...")
        asr = ASRClient()
        
        logger.info("Initializing TTS Client (Loading Qwen3-TTS)...")
        tts = TTSClient()
    except Exception as e:
        logger.error(f"Model Initialization Failed: {e}")
        return

    # 3. Test ASR
    logger.info("Testing ASR: Transcribing reference audio...")
    try:
        # ASRClient supports file path string directly
        text, lang = asr.transcribe(ref_audio_path)
        logger.info(f"ASR Result: [Text]: '{text}' [Lang]: '{lang}'")
        if not text:
            logger.warning("ASR returned empty text. This might be normal if the audio is silent, but check logs.")
    except Exception as e:
        logger.error(f"ASR Verification Failed: {e}")

    # 4. Test TTS
    target_text = "Phase two verification complete. The audio pipeline is operational."
    logger.info(f"Testing TTS: Synthesizing '{target_text}'...")
    try:
        # Generate
        audio_data, sr = tts.generate(target_text, language="English")
        
        if audio_data is not None:
            output_filename = "phase2_verify_output.wav"
            sf.write(output_filename, audio_data, sr)
            logger.info(f"TTS Result: Audio generated successfully.")
            logger.info(f"Saved verification output to: {os.path.abspath(output_filename)}")
        else:
            logger.error("TTS returned None.")
    except Exception as e:
        logger.error(f"TTS Verification Failed: {e}")

    print("-" * 50)
    logger.info("Verification Process Finished")
    print("-" * 50)

if __name__ == "__main__":
    main()
