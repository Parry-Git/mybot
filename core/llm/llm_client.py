from abc import ABC, abstractmethod
from typing import Generator, List, Dict
import os
import sys
from threading import Thread

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from openai import OpenAI

class LLMInterface(ABC):
    """Abstract interface for LLM clients."""
    @abstractmethod
    def stream_chat(self, history: List[Dict[str, str]], system_prompt: str = None) -> Generator[str, None, None]:
        pass

class OpenAILLMClient(LLMInterface):
    """API-based LLM Client using the OpenAI SDK format (e.g. for DeepSeek)."""
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        self.api_key = api_key or getattr(Config, 'DEEPSEEK_API_KEY', None)
        if not self.api_key:
             self.api_key = os.environ.get('DEEPSEEK_API_KEY')
        
        if not self.api_key:
             raise ValueError("API Key for DeepSeek is not set in Config or environment.")

        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def stream_chat(self, history: List[Dict[str, str]], system_prompt: str = "You are a helpful assistant") -> Generator[str, None, None]:
        """
        Streams response from the LLM.
        
        Args:
            history: List of message dicts [{"role": "user/assistant", "content": "..."}]
            system_prompt: Optional system prompt to prepend.
        
        Yields:
            Text chunks (tokens) as they are generated.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error during API call: {e}")
            yield ""

class LocalLLMClient(LLMInterface):
    """Local LLM Client using transformers and TextIteratorStreamer."""
    def __init__(self, model_id: str = "Qwen/Qwen3-1.7B-FP8"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import torch
        except ImportError:
            raise ImportError("Please install transformers and torch to use LocalLLMClient.")
            
        self.model_id = model_id
        self.device = Config.DEVICE
        self.dtype = Config.DTYPE
        
        print(f"Loading Local LLM model: {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        self.streamer_class = TextIteratorStreamer
        print("Local LLM Model loaded successfully.")

    def stream_chat(self, history: List[Dict[str, str]], system_prompt: str = "You are a helpful assistant") -> Generator[str, None, None]:
        messages = []
        if system_prompt:
             messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        streamer = self.streamer_class(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.8,
            temperature=0.7
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
