from __future__ import annotations

from typing import Iterable
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # pad the smaller
        m = max(a.shape[0], b.shape[0])
        a = np.pad(a, (0, m - a.shape[0]))
        b = np.pad(b, (0, m - b.shape[0]))
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom) 
