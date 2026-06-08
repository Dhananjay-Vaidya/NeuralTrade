"""Compatibility package so `python -m neuraltrade...` works from repo root.

The project uses a src-layout package in `src/neuraltrade`. This shim extends
the package search path so the package can also be executed without requiring
an editable install first.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "neuraltrade"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

