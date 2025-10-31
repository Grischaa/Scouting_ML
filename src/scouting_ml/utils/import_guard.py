# src/scouting_ml/utils/import_guard.py
"""
Ensures that any script under src/scouting_ml/* can run both
 - as a module (python -m scouting_ml.xyz)
 - or directly from VS Code (python xyz.py)
without breaking imports.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # points to .../Scout_Pred/src
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
