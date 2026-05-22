import sys
from pathlib import Path

# Add cmd/ so `ml_app` and `clientML` are importable as top-level packages.
# Add the flaps root so `scripts` is importable as a namespace package.
# Both entries are needed when pytest is invoked from the repository root.
_flaps_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_flaps_root / "cmd"))
sys.path.insert(0, str(_flaps_root))

