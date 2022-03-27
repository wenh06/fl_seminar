"""
dataset readers, for datasets from LIBSVM library
"""
from typing import Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file


__all__ = [
    "libsvmread",
]


def libsvmread(
    fp: str, multilabel: bool = False, toarray: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    features, labels = load_svmlight_file(fp, multilabel=multilabel, dtype=np.float32)
    if toarray:
        features = features.toarray()
    return features, labels
