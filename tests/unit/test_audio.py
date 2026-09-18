import threading
import time

import pytest

from core.audio.player import AudioPlayer
from core.config import AudioSettings
from core.tts.native import _PCMQueue
from core.types import AudioPacket


def test_large_packets_respect_playback_bound_and_cancel() -> None:
    player = AudioPlayer(settings=AudioSettings(playback_queue_seconds=0.02))
    player.start = lambda: None
    cancel = threading.Event()
    result = []
    packet = AudioPacket(b"\1\0" * 24000, 24000)
    producer = threading.Thread(target=lambda: result.append(player.enqueue(packet, cancel)))
    producer.start()
    time.sleep(0.05)
    assert player._queued_bytes == 960
    cancel.set()
    producer.join(timeout=1)
    assert result == [False]
    player.close()


def test_callback_consumes_pcm_and_fires_once() -> None:
    player = AudioPlayer()
    player.start = lambda: None
    callbacks = []
    assert player.enqueue(
        AudioPacket(b"\1\0" * 240, 24000), threading.Event(), lambda: callbacks.append(1)
    )
    output = bytearray(480)
    player._callback(output, 240, None, None)
    assert output == b"\1\0" * 240
    player._callback(output, 240, None, None)
    assert output == bytes(480)
    assert callbacks == [1]
    player.close()


def test_native_queue_pauses_producer_and_cancel_unblocks_it() -> None:
    paused = threading.Event()
    resumed = threading.Event()
    queue = _PCMQueue(paused.set, resumed.set)
    producer = threading.Thread(target=lambda: queue.put(bytes(960 * 100)))
    producer.start()
    assert paused.wait(timeout=1)
    assert queue.qsize() == 12
    queue.cancel.set()
    producer.join(timeout=1)
    assert not producer.is_alive()
    assert resumed.is_set()


def test_invalid_pcm_is_rejected() -> None:
    player = AudioPlayer(settings=AudioSettings(playback_enabled=False))
    with pytest.raises(ValueError, match="complete samples"):
        player.enqueue(AudioPacket(b"x", 24000), threading.Event())
