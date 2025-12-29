# espnet/__init__.py
"""
Shim package to expose vendored ESPNet.

The project vendors ESPNet under auto_avsr/espnet (and a duplicate under
lifelens/commpanion/app/auto_avsr/espnet). This shim makes `import espnet`
resolve correctly without pip-installing espnet.
"""

from pathlib import Path
import pkgutil

# Treat this as a namespace package
__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore

# Primary vendored ESPNet
vendored = Path(__file__).resolve().parent.parent / "auto_avsr" / "espnet"
if vendored.exists():
    __path__.append(str(vendored))  # type: ignore

# Secondary vendored ESPNet (duplicate tree)
vendored2 = (
    Path(__file__).resolve().parent.parent
    / "lifelens"
    / "commpanion"
    / "app"
    / "auto_avsr"
    / "espnet"
)
if vendored2.exists():
    __path__.append(str(vendored2))  # type: ignore
