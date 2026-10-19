"""Light entry point: ``python -m logic.controllers.cloud [--root DIR] [--json] <command> ...``.

The same commands as ``python main.py cloud``, without ``main.py``'s start-up
imports (torch and the Hydra stack): about 0.3 s and 35 MB per call instead of
2 s and 690 MB, which matters for the Studio's status polling.
"""

from __future__ import annotations

import argparse
import sys

from logic.controllers.cli.cloud_parser import add_cloud_args
from logic.controllers.cloud.cli import run_cloud


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m logic.controllers.cloud", description=__doc__)
    add_cloud_args(parser)
    opts = vars(parser.parse_args(argv))
    try:
        run_cloud(opts)
    except Exception as exc:  # noqa: BLE001 - one clean error line for the Studio and scripts
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
