"""Local launcher for the CALIBER Kedro project."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_DIRECTORY = PROJECT_ROOT / "runtime_tmp"
TEMP_DIRECTORY.mkdir(exist_ok=True)

os.environ.setdefault("TEMP", str(TEMP_DIRECTORY))
os.environ.setdefault("TMP", str(TEMP_DIRECTORY))
os.environ.setdefault("KEDRO_DISABLE_TELEMETRY", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(PROJECT_ROOT / ".vendor"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    """Delegate commands to the Kedro CLI with local dependencies enabled."""
    from kedro.framework.cli import main as kedro_main

    kedro_main()


if __name__ == "__main__":
    main()
