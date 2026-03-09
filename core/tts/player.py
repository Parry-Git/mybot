import os
import logging
import scipy.io.wavfile as wavfile
import numpy as np
import subprocess
import time
from bot.core.utils.config import Config

logger = logging.getLogger(__name__)

class AudioPlayer:
    def __init__(self, rate=Config.TTS_SAMPLE_RATE):
        self.rate = rate
        self.temp_file = "response_temp.wav"
        self._process = None # To hold the aplay process

    def is_playing(self):
        """Check if the aplay process is still running."""
        if self._process is not None:
            # poll() returns None if process is still running
            if self._process.poll() is None:
                return True
            else:
                self._process = None
        return False

    def start(self):
        logger.info("Audio player initialized (Async System-call mode).")

    def stop_playback(self):
        """Forcibly stop the current audio playback."""
        if self._process and self._process.poll() is None:
            logger.info("Interrupting playback...")
            self._process.terminate()
            self._process.wait()
        self._process = None

    def stop(self):
        self.stop_playback()
        if os.path.exists(self.temp_file):
            try: os.remove(self.temp_file)
            except: pass

    def play(self, audio_data, sample_rate=None):
        """
        Plays audio in background using aplay.
        """
        try:
            self.stop_playback() # Stop previous if still playing
            
            rate = sample_rate if sample_rate is not None else self.rate
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(self.temp_file, rate, audio_int16)
            
            # Use Popen instead of run to make it non-blocking
            self._process = subprocess.Popen(
                ["aplay", "-q", self.temp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"Playback started in background.")
        except Exception as e:
            logger.error(f"Playback error: {e}")
