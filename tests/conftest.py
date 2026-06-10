"""Test path bootstrap.

The legacy ``tests/test_counting.py`` imports the implementation module as
``from contar import ...`` (bare), which only resolves when the package
directory ``contador_pajaros/`` is on ``sys.path``. The newer
``test_count_vehicles_linecrossing.py`` imports ``contador_pajaros.contar``,
which needs the project root on ``sys.path``.

Adding both here lets ``python -m pytest`` work from the repo root with NO
``PYTHONPATH`` env var and WITHOUT an editable install (which would pull in the
heavy cv2/ultralytics deps). The headless logic under test lazy-imports those,
so the pure-logic suite runs on a plain Python + pytest.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PKG = _ROOT / "contador_pajaros"

for _p in (str(_ROOT), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
