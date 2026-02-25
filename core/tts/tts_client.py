import torch
import logging
import os
import gc
from qwen_tts import Qwen3TTSModel
from bot.core.utils.config import Config

logger = logging.getLogger(__name__)

class TTSClient:
    def __init__(self, model_path=Config.TTS_MODEL_PATH, device=Config.DEVICE, dtype=Config.DTYPE):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.prompt_ids = None
        
        logger.info(f"Loading TTS model from {model_path} on {device} with {dtype}")
        try:
            # Reverting to device_map in from_pretrained which is the standard way for these wrappers
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                dtype=self.dtype,
                device_map=self.device, 
                attn_implementation=Config.ATTN_IMPLEMENTATION
            )
            # Ensure the underlying torch model is in eval mode
            if hasattr(self.model, 'model'):
                self.model.model.eval()
            
            logger.info("TTS model loaded successfully.")
            
            # Determine if we need to load a reference prompt (for Base models)
            if "Base" in model_path:
                self._init_base_prompt()
                
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def _init_base_prompt(self):
        ref_audio_path = Config.TTS_REF_AUDIO_PATH
        if not os.path.exists(ref_audio_path):
             # Try absolute path fallback
            ref_audio_path = os.path.join(os.getcwd(), Config.TTS_REF_AUDIO_PATH)

        if not os.path.exists(ref_audio_path):
            logger.warning(f"Reference audio not found.")
            return

        logger.info(f"Encoding reference voice: {ref_audio_path}")
        
        try:
            # Create voice clone prompt
            self.prompt_ids = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                x_vector_only_mode=True
            )
            logger.info("Voice clone prompt ready.")
        except Exception as e:
            logger.error(f"Voice clone encoding error: {e}")

    def generate(self, text, language="Chinese"):
        """
        Generate audio from text.
        """
        try:
            # Clear CUDA cache before heavy generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # The wrapper's generate methods handle torch.no_grad() internally usually, 
            # but wrapping it here is safer.
            with torch.no_grad():
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=self.prompt_ids
                )
            
            if wavs:
                return wavs[0], sr
            return None, sr
            
        except Exception as e:
            logger.error(f"TTS Generation error: {e}")
            return None, Config.TTS_SAMPLE_RATE
        finally:
            # Trigger garbage collection
            gc.collect()
