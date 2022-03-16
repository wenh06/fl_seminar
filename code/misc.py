"""
"""

import pathlib
from functools import wraps
from typing import List, Any, Callable


__all__ = [
    "PROJECT_DIR", "BUILTIN_DATA_DIR", "CACHED_DATA_DIR", "LOG_DIR",
    "default_class_repr", "ReprMixin",
    "isclass",
    "experiment_indicator",
]


PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent

BUILTIN_DATA_DIR = PROJECT_DIR / "data"

CACHED_DATA_DIR = PROJECT_DIR / ".data_cache"

LOG_DIR = PROJECT_DIR / ".logs"


CACHED_DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def default_class_repr(c:object, align:str="center", depth:int=1) -> str:
    """ finished, checked,

    Parameters
    ----------
    c: object,
        the object to be represented
    align: str, default "center",
        the alignment of the class arguments

    Returns
    -------
    str,
        the representation of the class
    """
    indent = 4*depth*" "
    closing_indent = 4*(depth-1)*" "
    if not hasattr(c, "extra_repr_keys"):
        return repr(c)
    elif len(c.extra_repr_keys()) > 0:
        max_len = max([len(k) for k in c.extra_repr_keys()])
        extra_str = "(\n" + \
            ",\n".join([
                f"""{indent}{k.ljust(max_len, " ") if align.lower() in ["center", "c"] else k} = {default_class_repr(eval(f"c.{k}"),align,depth+1)}""" \
                    for k in c.__dir__() if k in c.extra_repr_keys()
                ]) + \
            f"{closing_indent}\n)"
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


def isclass(obj:Any) -> bool:
    """ finished, checked,

    Parameters
    ----------
    obj: any object,
        any object, including class, instance of class, etc

    Returns
    -------
    bool:
        True if `obj` is a class, False otherwise
    """
    try:
        return issubclass(obj, object)
    except TypeError:
        return False


def experiment_indicator(name:str) -> Callable:
    """
    """
    def decorator(func:Callable) -> Callable:
        @wraps(func)
        def wrapper(*args:Any, **kwargs:Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start experiment {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End experiment {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
        return wrapper
    return decorator
