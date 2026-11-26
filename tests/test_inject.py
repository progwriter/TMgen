import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy
import pytest

from tmgen import TrafficMatrix
from tmgen.inject import ditg_injector


class _FakeProcess:
    started: list[list[str]] = []

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.started.append(command)
        self.signals: list[int] = []
        self.terminated = False

    def poll(self) -> int:
        return 0

    def wait(self) -> int:
        return 0

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture
def fake_processes(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    _FakeProcess.started = []
    monkeypatch.setattr(ditg_injector, "Popen", _FakeProcess)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    return _FakeProcess.started


def test_main_starts_a_receiver_and_a_sender_per_destination_and_epoch(
    tmp_path: Path, fake_processes: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tm.npz"
    TrafficMatrix(numpy.full((2, 2, 2), 2.2)).save(path)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: None)
    ditg_injector.main(
        [
            *("--tm", str(path), "--epoch-length", "3", "--id", "0"),
            *("--destinations", '{"1": ["10.0.0.2", "10.0.0.3"]}', "--port", "9000"),
            *("--scale", "2"),
        ]
    )
    receivers = [c for c in fake_processes if c[0] == "ITGRecv"]
    senders = [c for c in fake_processes if c[0] == "ITGSend"]
    assert len(receivers) == 2
    assert len(senders) == 4
    assert senders[0][senders[0].index("-t") + 1] == "3000.0"
    assert senders[0][senders[0].index("-C") + 1] == "5"  # ceil(2.2 * 2)
    assert senders[0][senders[0].index("-a") + 1] == "10.0.0.2"


def test_importing_the_module_leaves_ctrl_c_alone() -> None:
    code = (
        "import signal, tmgen.inject.ditg_injector;"
        "print(signal.getsignal(signal.SIGINT) is signal.default_int_handler)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True"


def test_stop_terminates_the_running_processes(
    tmp_path: Path, fake_processes: list[list[str]]
) -> None:
    path = tmp_path / "tm.npz"
    TrafficMatrix(numpy.ones((2, 2, 1))).save(path)
    injector = ditg_injector.DITGinjector(path, 1.0, 0, {"1": ["10.0.0.2"]}, 9000, 1)
    receiver = injector._start_receiver()
    injector._start_senders(0)
    injector.stop()
    assert receiver.terminated  # type: ignore[attr-defined]
    assert all(p.terminated for p in injector._send_processes)  # type: ignore[attr-defined]
