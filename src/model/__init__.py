"""Cat Image Arena feature pipeline (CPU-only).

Public entry points are re-exported here for notebook use::

    import sys; sys.path.insert(0, "src")
    from model import extract, features, paths

The modules are intentionally small and free of side effects on import so that
``scripts/*.py`` stay thin command-line wrappers.
"""

from __future__ import annotations

import os

from . import paths as _paths

# Cache directories must stay inside the project: the shared ~/.cache may not be
# writable on this machine (restricted sandbox), which would break the pretrained
# weight download and matplotlib's font cache.  ``setdefault`` keeps user
# overrides working.
os.environ.setdefault("TORCH_HOME", str(_paths.ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(_paths.ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_paths.ROOT / ".cache" / "xdg"))

__version__ = "1.0.0"
