import fcntl
import os

from scripts import service


def test_stale_pid_file_is_not_treated_as_running(tmp_path, monkeypatch):
    path = tmp_path / "mybot.lock"
    path.write_text("999999")
    monkeypatch.setattr(service, "LOCK", path)
    assert service.active_pid() is None


def test_only_live_lock_owner_is_reported(tmp_path, monkeypatch):
    path = tmp_path / "mybot.lock"
    monkeypatch.setattr(service, "LOCK", path)
    with path.open("w+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        assert service.active_pid() == os.getpid()
    assert service.active_pid() is None
