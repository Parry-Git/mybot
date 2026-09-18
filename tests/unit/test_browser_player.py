import struct
import threading
import time

from core.audio.browser import BrowserAudioPlayer
from core.types import AudioPacket


def test_browser_playback_waits_for_render_acknowledgement():
    player = BrowserAudioPlayer()
    assert player.attach()
    assert not player.attach()
    generation = player.outgoing.get_nowait()["value"]
    callbacks = []
    assert player.enqueue(
        AudioPacket(bytes(960), 24000), threading.Event(), lambda: callbacks.append(1)
    )
    data = player.outgoing.get_nowait()
    assert struct.unpack("<I", data[:4])[0] == generation
    assert player.is_playing()
    assert callbacks == []
    player.acknowledge(generation, 480)
    assert callbacks == [1]
    assert not player.is_playing()
    player.close()


def test_clear_ignores_stale_browser_audio_ack():
    player = BrowserAudioPlayer()
    player.attach()
    old = player.outgoing.get_nowait()["value"]
    player.enqueue(AudioPacket(bytes(960), 24000), threading.Event())
    player.clear()
    current = player.outgoing.get_nowait()["value"]
    assert current != old
    player.enqueue(AudioPacket(bytes(960), 24000), threading.Event())
    player.acknowledge(old, 480)
    assert player.is_playing()
    player.acknowledge(current, 480)
    assert not player.is_playing()
    player.close()


def test_disconnect_unblocks_browser_backpressure():
    player = BrowserAudioPlayer(queue_seconds=0.02)
    player.attach()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            player.enqueue(AudioPacket(bytes(9600), 24000), threading.Event())
        )
    )
    thread.start()
    time.sleep(0.05)
    assert player._pending == 960
    player.detach()
    thread.join(timeout=1)
    assert result == [False]
    assert not thread.is_alive()
    player.close()
