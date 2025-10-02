from __future__ import annotations

import re

from pandas.compat._optional import import_optional_dependency

ne = import_optional_dependency("numexpr", errors="warn")
NUMEXPR_INSTALLED = ne is not None
if ne is None:
    # use tuple to make types consistent with mypy
    NUMEXPR_VERSION = (0, 0, 0)
else:
    NUMEXPR_VERSION = tuple(map(int, re.findall(r"(\d+)", ne.__version__)))
