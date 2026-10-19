"""Run a command and write its exit code to a file (used by the local backend).

Usage: ``python -m logic.controllers.cloud.local._wrap EXIT_FILE -- COMMAND...``.
SIGTERM is passed on to the command's process group, and the exit code is still written.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def main(argv: list) -> int:
    exit_file, separator, *command = argv
    if separator != "--" or not command:
        raise SystemExit("usage: _wrap EXIT_FILE -- COMMAND...")
    proc = subprocess.Popen(command)

    def forward(signum: int, _frame: object) -> None:
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = proc.wait()
    tmp = Path(exit_file + ".tmp")
    tmp.write_text(str(code if code >= 0 else 128 - code))
    os.replace(tmp, exit_file)
    return code


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) == 0 else 1)
