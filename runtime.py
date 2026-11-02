"""Runtime path configuration for the CLIP-HUGS integration."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
CLIPGAUSSIAN_ROOT = REPOSITORY_ROOT / "clipgaussian_core"


def configure_runtime() -> Path | None:
    """Make local CLIP code and an external ML-HUGS checkout importable.

    ML-HUGS can either be installed in the active environment or exposed with
    ``ML_HUGS_ROOT``. The latter must point to a checkout containing ``hugs/``.
    """
    if str(CLIPGAUSSIAN_ROOT) not in sys.path:
        sys.path.insert(0, str(CLIPGAUSSIAN_ROOT))

    ml_hugs_root = os.environ.get("ML_HUGS_ROOT")
    if ml_hugs_root:
        root = Path(ml_hugs_root).expanduser().resolve()
        if not (root / "hugs").is_dir():
            raise RuntimeError(
                f"ML_HUGS_ROOT must contain a hugs/ package: {root}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    if importlib.util.find_spec("hugs") is None:
        raise RuntimeError(
            "ML-HUGS is not importable. Install it in the active environment "
            "or set ML_HUGS_ROOT to an ML-HUGS checkout."
        )
    return None


def configure_clipgaussian() -> Path:
    """Make the external CLIPGaussian 3D Gaussian-splatting code importable."""
    value = os.environ.get("CLIPGAUSSIAN_ROOT")
    if not value:
        raise RuntimeError(
            "Set CLIPGAUSSIAN_ROOT to the CLIPGaussian/3D checkout for the "
            "optional PLY pipeline."
        )
    root = Path(value).expanduser().resolve()
    required = (root / "models" / "gs", root / "arguments")
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "CLIPGAUSSIAN_ROOT must contain models/gs and arguments: "
            f"{root}"
        )
    for path in (root, root / "models" / "gs"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


def require_environment_path(name: str) -> Path:
    """Return a required environment path with a useful error message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to the required external path.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"{name} does not exist: {path}")
    return path
