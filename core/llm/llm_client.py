from __future__ import annotations

import asyncio
import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from concurrent.futures import Future
from typing import Optional

import httpx
from openai import AsyncOpenAI

from core.config import LLMSettings

logger = logging.getLogger(__name__)
Message = dict[str, str]


class LLMInterface(ABC):
    @abstractmethod
    def stream_chat(
        self,
        history: Sequence[Message],
        system_prompt: Optional[str] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        raise NotImplementedError


class OpenAILLMClient(LLMInterface):
    """Streaming OpenAI-compatible client, configured through environment variables."""

    def __init__(self, settings: Optional[LLMSettings] = None) -> None:
        self.settings = settings or LLMSettings()
        if not self.settings.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self.client = AsyncOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=httpx.Timeout(self.settings.timeout_seconds, connect=10.0),
            max_retries=0,
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="llm-http", daemon=True)
        self._thread.start()
        self._state_lock = threading.RLock()
        self._active: set[Future] = set()
        self._closed = False

    def stream_chat(
        self,
        history: Sequence[Message],
        system_prompt: Optional[str] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        if cancel is not None and cancel.is_set():
            return
        prompt = self.settings.system_prompt if system_prompt is None else system_prompt
        messages: list[Message] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.extend(dict(message) for message in history)

        extra_body = None
        if "deepseek" in self.settings.base_url.lower():
            thinking_type = "enabled" if self.settings.thinking_enabled else "disabled"
            extra_body = {"thinking": {"type": thinking_type}}
        tokens: queue.Queue[str] = queue.Queue(maxsize=32)

        async def receive() -> None:
            response = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                stream=True,
                max_tokens=self.settings.max_output_tokens,
                extra_body=extra_body,
            )
            try:
                async for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        while True:
                            try:
                                tokens.put_nowait(content)
                                break
                            except queue.Full:
                                await asyncio.sleep(0.01)
            finally:
                await response.close()

        def forget(future: Future) -> None:
            with self._state_lock:
                self._active.discard(future)

        with self._state_lock:
            if self._closed:
                raise RuntimeError("LLM client is closed")
            future = asyncio.run_coroutine_threadsafe(receive(), self._loop)
            self._active.add(future)
            future.add_done_callback(forget)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    return
                try:
                    yield tokens.get(timeout=0.02)
                except queue.Empty:
                    if future.done():
                        future.result()
                        return
        finally:
            future.cancel()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            for future in list(self._active):
                future.cancel()

        async def shutdown() -> None:
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.client.close()

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout=5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if not self._thread.is_alive():
                self._loop.close()
