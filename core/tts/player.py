import pyaudio
import threading
import queue
import logging
import numpy as np
from bot.core.utils.config import Config

logger = logging.getLogger(__name__)

class AudioPlayer:
    def __init__(self, rate=Config.TTS_SAMPLE_RATE):
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._pyaudio = pyaudio.PyAudio()
        self._stream = None
        self.rate = rate

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        
        try:
            # Output in Float32
            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.rate,
                output=True
            )
            logger.info(f"Audio player started at {self.rate}Hz.")
        except Exception as e:
            logger.error(f"Failed to open audio output stream: {e}")
            return

        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if self._pyaudio:
            self._pyaudio.terminate()
        
        logger.info("Audio player stopped.")

    def play(self, audio_data):
        """
        Add audio data to the queue.
        audio_data: np.ndarray (float32)
        """
        self.queue.put(audio_data)

    def _play_loop(self):
        while not self._stop_event.is_set():
            try:
                audio_data = self.queue.get(timeout=0.1)
                
                # Normalize / Check format
                if isinstance(audio_data, np.ndarray):
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    data_bytes = audio_data.tobytes()
                else:
                    data_bytes = audio_data
                
                if self._stream:
                    self._stream.write(data_bytes)
                
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in playback loop: {e}")
