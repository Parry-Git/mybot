import re
from typing import Generator, Iterator

class TokenAggregator:
    """
    Aggregates a stream of tokens into chunks suitable for TTS processing.
    It filters out <think>...</think> reasoning blocks and chunks the output
    based on punctuation (phrase-level streaming).
    """
    def __init__(self, chunk_length_threshold: int = 12):
        self.chunk_length_threshold = chunk_length_threshold
        # Punctuation that triggers a chunk release
        self.punctuation_pattern = re.compile(r'([。！？，；、\!\?\,])')
        
    def aggregate(self, token_stream: Iterator[str]) -> Generator[str, None, None]:
        """
        Takes an iterator of tokens and yields sentences/phrases.
        """
        buffer = ""
        in_think_block = False
        think_buffer = ""
        
        for token in token_stream:
            # Handle <think> blocks across tokens
            if in_think_block:
                think_buffer += token
                if "</think>" in think_buffer:
                    # Find where </think> ends and keep the rest
                    parts = think_buffer.split("</think>")
                    in_think_block = False
                    buffer += parts[-1] # Append anything after </think> to the main buffer
                    think_buffer = ""
                continue
            
            # Not in think block
            buffer += token
            
            # Check if we just entered a think block
            if "<think>" in buffer:
                parts = buffer.split("<think>")
                buffer = parts[0]
                in_think_block = True
                think_buffer = parts[1] if len(parts) > 1 else ""
                
                # Yield anything before <think> if there's any valid chunk
                if buffer.strip():
                    yield buffer.strip()
                    buffer = ""
                continue
                
            # If we might be in the middle of receiving "<think>" (e.g., "<th")
            # We delay yielding if the buffer ends with something that could be a tag
            if buffer.endswith("<") or buffer.endswith("<t") or buffer.endswith("<th") or buffer.endswith("<thi") or buffer.endswith("<thin") or buffer.endswith("<think"):
                continue

            # Check for punctuation to chunk
            match = self.punctuation_pattern.search(buffer)
            if match:
                # Find the last punctuation index
                # We want to split at the *first* punctuation we see to keep chunks small and responsive
                # But to be safe, let's just split at the first punctuation
                first_punct_idx = match.end()
                chunk = buffer[:first_punct_idx]
                buffer = buffer[first_punct_idx:]
                
                if chunk.strip():
                    yield chunk.strip()
            
            # Or if buffer is getting too long even without punctuation, maybe force chunk
            # But usually it's better to wait for punctuation for natural TTS.
            
        # Yield remaining buffer
        if buffer.strip():
            yield buffer.strip()
