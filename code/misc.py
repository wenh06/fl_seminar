"""
"""

import os, sys


__all__ = [
    "PROJECT_DIR", "BUILTIN_DATA_DIR", "CACHED_DATA_DIR",
    "default_class_repr",
]


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUILTIN_DATA_DIR = os.path.join(PROJECT_DIR, "data")

CACHED_DATA_DIR = os.path.join(PROJECT_DIR, ".data_cache")


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
