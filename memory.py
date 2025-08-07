from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np

from .dsl import State, compute_structural_fingerprint


@dataclass
class FingerprintStore:
    entries: List[Tuple[np.ndarray, Dict[str, Any]]] = field(default_factory=list)

    def add(self, state: State, meta: Dict[str, Any]) -> None:
        self.entries.append((compute_structural_fingerprint(state), dict(meta)))

    def nearest(self, state: State, k: int = 5) -> List[Dict[str, Any]]:
        from .utils import cosine_similarity
        fp = compute_structural_fingerprint(state)
        sims = [(cosine_similarity(fp, e[0]), e[1]) for e in self.entries]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in sims[:k]]


@dataclass
class NarrativeLog:
    events: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, event: Dict[str, Any]) -> None:
        self.events.append(dict(event))

    def summarize(self) -> Dict[str, Any]:
        # Very simple stats
        counts: Dict[str, int] = {}
        for e in self.events:
            k = e.get("type", "unknown")
            counts[k] = counts.get(k, 0) + 1
        return counts 
