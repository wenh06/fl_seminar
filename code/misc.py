"""
"""

import pathlib
from typing import List


__all__ = [
    "PROJECT_DIR", "BUILTIN_DATA_DIR", "CACHED_DATA_DIR", "LOG_DIR",
    "default_class_repr", "ReprMixin",
]


PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent

BUILTIN_DATA_DIR = PROJECT_DIR / "data"

CACHED_DATA_DIR = PROJECT_DIR / ".data_cache"

LOG_DIR = PROJECT_DIR / ".logs"


CACHED_DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def default_class_repr(c:object, align:str="center") -> str:
    """
    """
    if hasattr(c, "extra_repr_keys") and len(c.extra_repr_keys()) > 0:
        max_len = max([len(k) for k in c.extra_repr_keys()])
        extra_str = "(\n" + \
            ",\n".join([
                f"""    {k.ljust(max_len, " ") if align.lower() in ["center", "c"] else k} = {v}""" \
                    for k,v in c.__dict__.items() if k in c.extra_repr_keys()
                ]) + \
            "\n)"
    else:
        extra_str = ""
    return f"{c.__class__.__name__}{extra_str}"


class ReprMixin(object):
    """
    Mixin for enhanced __repr__ and __str__.
    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return []
