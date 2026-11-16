"""Canonical entry point for formal training of all learned E1 methods."""

from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from comparison_runners.train_drl_baselines import main


if __name__ == "__main__":
    main()
