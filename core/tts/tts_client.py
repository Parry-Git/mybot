import torch
import logging
import os
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
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=Config.ATTN_IMPLEMENTATION
            )
            logger.info("TTS model loaded.")
            
            # Determine if we need to load a reference prompt (for Base models)
            if "Base" in model_path:
                self._init_base_prompt()
                
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def _init_base_prompt(self):
        # We look for a reference audio file
        ref_audio_path = Config.TTS_REF_AUDIO_PATH
        if not os.path.exists(ref_audio_path):
            # Try absolute path based on workspace
            ref_audio_path = os.path.join(os.getcwd(), Config.TTS_REF_AUDIO_PATH)
            if not os.path.exists(ref_audio_path):
                logger.warning(f"Reference audio not found at {ref_audio_path}. Voice cloning might fail.")
                return

        logger.info(f"Loading reference audio from {ref_audio_path}")
        
        try:
            # We use x_vector_only_mode=True to avoid needing transcript for the reference audio
            # This is safer for generic usage.
            self.prompt_ids = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                x_vector_only_mode=True
            )
            logger.info("Voice clone prompt created.")
        except Exception as e:
            logger.error(f"Failed to create voice clone prompt: {e}")

    def generate(self, text, language="Chinese"):
        """
        Generate audio from text.
        
        Args:
            text (str): Text to synthesize.
            language (str): Language code/name.
            
        Returns:
            np.ndarray: Audio data (float32)
            int: Sample rate
        """
        try:
            wavs = None
            sr = Config.TTS_SAMPLE_RATE
            
            if "Base" in self.model_path:
                if self.prompt_ids is None:
                    logger.error("No voice clone prompt available.")
                    return None, sr
                
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=self.prompt_ids
                )
            else:
                # CustomVoice or VoiceDesign
                speakers = self.model.get_supported_speakers()
                speaker = speakers[0] if speakers else "Vivian"
                
                # If VoiceDesign, API is different (generate_voice_design)
                # But Config says CustomVoice or Base usually. 
                # Let's check model type from model config or path string.
                if "VoiceDesign" in self.model_path:
                     # VoiceDesign requires 'instruct'
                     wavs, sr = self.model.generate_voice_design(
                        text=text,
                        language=language,
                        instruct="Speak naturally."
                     )
                else:
                    wavs, sr = self.model.generate_custom_voice(
                        text=text,
                        language=language,
                        speaker=speaker
                    )
            
            if wavs:
                return wavs[0], sr
            return None, sr
            
        except Exception as e:
            logger.error(f"TTS Generation error: {e}")
            return None, Config.TTS_SAMPLE_RATE
