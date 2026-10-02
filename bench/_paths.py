"""Makes `scripts/` importable from `bench/` and exposes the shared utilities.

The synthetic data generator lives in ``scripts/compare_baseline.py`` (where it
is also used by ``bench_numpy.py``); it is reused here rather than duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compare_baseline import correlation, lambda_min, make_data  # noqa: E402,F401

__all__ = ["correlation", "lambda_min", "make_data", "SCRIPTS_DIR"]
