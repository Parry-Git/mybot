from kokoro import KPipeline
import torch
import numpy as np

class KokoroTTSClient:
    def __init__(self, lang_code='a', device='cuda'):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.device = device

    def generate(self, text, voice='af_heart', speed=1.0):
        """
        Generates audio from text using the Kokoro TTS model.

        Args:
            text (str): The text to synthesize.
            voice (str, optional): The voice to use. Defaults to 'af_heart'.
            speed (float, optional): The speaking speed. Defaults to 1.0.

        Returns:
            np.ndarray: The generated audio as a numpy array.
        """
        full_audio = []
        generator = self.pipeline(text, voice=voice, speed=speed)
        for i, (gs, ps, audio) in enumerate(generator):
            full_audio.append(audio)
        
        if not full_audio:
            return np.array([], dtype=np.float32)

        return np.concatenate(full_audio)

    def get_sample_rate(self):
        return 24000
