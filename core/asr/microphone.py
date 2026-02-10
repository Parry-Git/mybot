import pyaudio
import logging
import os
import sys
from contextlib import contextmanager
from bot.core.utils.config import Config

logger = logging.getLogger(__name__)

@contextmanager
def no_alsa_err():
    """
    Context manager to suppress ALSA configuration errors by redirecting stderr to /dev/null.
    Useful for PyAudio on WSL/Linux where ALSA complains about missing hardware.
    """
    try:
        # Open /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Save the original stderr file descriptor
        old_stderr = os.dup(2)
        # Flush the python-level stderr
        sys.stderr.flush()
        # Redirect stderr to /dev/null
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except Exception:
        # If anything goes wrong with redirection, just execute the code
        yield
    finally:
        try:
            # Restore stderr
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

class Microphone:
    """
    A context manager wrapper around PyAudio for recording audio.
    Yields raw audio chunks compatible with VAD requirements.
    """
    def __init__(self):
        # Suppress ALSA noise during initialization
        with no_alsa_err():
            self._pyaudio = pyaudio.PyAudio()
            
        self._stream = None
        self._chunk_size = Config.VAD_FRAME_SAMPLES

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self._stream is not None:
            return

        try:
            self._stream = self._pyaudio.open(
                format=Config.FORMAT,
                channels=Config.CHANNELS,
                rate=Config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self._chunk_size,
            )
            logger.info("Microphone stream started.")
        except Exception as e:
            logger.error(f"Failed to start microphone stream: {e}")
            raise

    def stop(self):
        if self._stream is not None:
            if self._stream.is_active():
                self._stream.stop_stream()
            self._stream.close()
            self._stream = None
            logger.info("Microphone stream stopped.")

        if self._pyaudio is not None:
            self._pyaudio.terminate()
            # self._pyaudio = None  # PyAudio instance might be reusable, but here we terminate it for safety on exit
    
    def stream(self):
        """
        Generator that yields audio chunks.
        """
        if self._stream is None:
            raise RuntimeError("Microphone stream not started. Use with context manager or call start().")

        logger.info("Recording started... (Press Ctrl+C to stop in CLI)")
        try:
            while True:
                try:
                    # exception_on_overflow=False prevents crashes if processing is slow
                    data = self._stream.read(self._chunk_size, exception_on_overflow=False)
                    if len(data) == 0:
                        break
                    yield data
                except IOError as e:
                    logger.warning(f"Audio input overflow/error: {e}")
                    continue
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Recording generator finished.")
