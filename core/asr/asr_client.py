import torch
import numpy as np
import logging
from qwen_asr import Qwen3ASRModel
from bot.core.utils.config import Config

logger = logging.getLogger(__name__)

class ASRClient:
    def __init__(self, model_path=Config.ASR_MODEL_PATH, device=Config.DEVICE, dtype=Config.DTYPE):
        self.device = device
        self.dtype = dtype
        
        logger.info(f"Loading ASR model from {model_path} on {device} with {dtype}")
        try:
            self.model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=self.dtype,
                device_map=self.device,
            )
            logger.info("ASR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise

    def transcribe(self, audio_data, sample_rate=Config.SAMPLE_RATE):
        """
        Transcribe audio data.
        
        Args:
            audio_data: np.ndarray (float32, -1.0 to 1.0) or bytes (int16 PCM).
            sample_rate: int, default 16000.
            
        Returns:
            str: Transcribed text.
        """
        input_audio = None
        
        if isinstance(audio_data, bytes):
            # Convert int16 bytes to float32 numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            input_audio = (audio_np, sample_rate)
        elif isinstance(audio_data, np.ndarray):
            input_audio = (audio_data, sample_rate)
        else:
            # Assuming file path or URL
            input_audio = audio_data

        try:
            # Transcribe
            # Note: Qwen3ASRModel.transcribe returns a list of results
            results = self.model.transcribe(
                audio=input_audio,
                language=None, # Automatic language detection
                return_time_stamps=False,
            )
            
            if results and len(results) > 0:
                text = results[0].text
                lang = results[0].language
                logger.debug(f"ASR Output: {text} ({lang})")
                return text.strip(), lang
            return "", None
            
        except Exception as e:
            logger.error(f"ASR Transcription error: {e}")
            return "", None
