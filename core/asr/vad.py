import collections
import webrtcvad
import logging
from bot.core.utils.config import Config

logger = logging.getLogger(__name__)

class VadDetector:
    """
    Voice Activity Detector (VAD) using WebRTC.
    Filters audio stream and yields segments containing active speech.
    """
    def __init__(self):
        self.vad = webrtcvad.Vad(Config.VAD_MODE)
        self.sample_rate = Config.SAMPLE_RATE
        self.frame_duration_ms = Config.VAD_FRAME_MS
        
        # Buffers
        self._ring_buffer = collections.deque(maxlen=Config.SPEECH_TRIGGER_FRAMES)
        self._triggered = False
        self._voiced_frames = []
        self._silence_counter = 0

    def is_speech(self, frame_bytes):
        """Returns True if the frame contains speech."""
        try:
            return self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception as e:
            logger.error(f"VAD Error: {e}")
            return False

    def process_chunk(self, chunk):
        """
        Process a single audio chunk.
        
        Returns:
            bytes: A complete utterance (audio bytes) if a sentence just finished.
            None: If still processing or silence.
        """
        is_speech = self.is_speech(chunk)

        if not self._triggered:
            self._ring_buffer.append((chunk, is_speech))
            
            # Check if we should trigger (enough consecutive speech frames)
            num_voiced = len([f for f, s in self._ring_buffer if s])
            
            # Logic: If > 90% of the ring buffer is speech, trigger start
            if num_voiced >= Config.SPEECH_TRIGGER_FRAMES * 0.9:
                self._triggered = True
                self._silence_counter = 0
                logger.debug("VAD Triggered: Start of Speech")
                
                # Flush ring buffer to voiced frames to include the start of the sentence
                for f, s in self._ring_buffer:
                    self._voiced_frames.append(f)
                self._ring_buffer.clear()
        else:
            # We are in SPEECH state
            self._voiced_frames.append(chunk)
            
            if is_speech:
                self._silence_counter = 0
            else:
                self._silence_counter += 1
                
            # Check if we should release (enough consecutive silence frames)
            if self._silence_counter >= Config.SILENCE_LIMIT_FRAMES:
                logger.debug("VAD Released: End of Speech")
                self._triggered = False
                
                # Combine all frames
                full_audio = b''.join(self._voiced_frames)
                
                # Reset buffers
                self._voiced_frames = []
                self._ring_buffer.clear()
                self._silence_counter = 0
                
                return full_audio
                
        return None

    def reset(self):
        self._triggered = False
        self._voiced_frames = []
        self._ring_buffer.clear()
        self._silence_counter = 0
