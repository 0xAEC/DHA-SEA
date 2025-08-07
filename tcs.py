from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np

from .dsl import State, compute_structural_fingerprint


@dataclass
class Overlay:
    kind: str
    data: Any


@dataclass
class TCSHypothesis:
    hypothesis_id: str
    confidence: float
    state: State
    overlays: Dict[str, Overlay] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_distance: Optional[float] = None
    total_cost: float = 0.0
    cached_fingerprint: Optional[np.ndarray] = None

    @staticmethod
    def create(state: State, confidence: float) -> "TCSHypothesis":
        return TCSHypothesis(hypothesis_id=str(uuid.uuid4()), confidence=confidence, state=state)

    def fork(self, *, tag: str = "") -> "TCSHypothesis":
        forked = TCSHypothesis(
            hypothesis_id=f"{self.hypothesis_id}:{tag}:{str(uuid.uuid4())[:8]}",
            confidence=self.confidence,
            state=self.state.copy(),
            overlays=dict(self.overlays),
            history=list(self.history),
            last_distance=self.last_distance,
            total_cost=self.total_cost,
            cached_fingerprint=None,
        )
        return forked

    def add_overlay(self, kind: str, data: Any) -> None:
        self.overlays[kind] = Overlay(kind=kind, data=data)

    def get_overlay(self, kind: str) -> Optional[Overlay]:
        return self.overlays.get(kind)

    def fingerprint(self) -> np.ndarray:
        if self.cached_fingerprint is None:
            self.cached_fingerprint = compute_structural_fingerprint(self.state)
        return self.cached_fingerprint

    def record_action(self, operator_name: str, params: Dict[str, Any], info_gain: float, cost: float) -> None:
        self.history.append({
            "operator": operator_name,
            "params": dict(params),
            "ΔI": float(info_gain),
            "ΔC": float(cost),
        })
        self.total_cost += float(cost)

    def set_progress(self, distance_to_goal: float) -> None:
        self.last_distance = distance_to_goal

    def information_gain(self, new_distance: float) -> float:
        if self.last_distance is None:
            return 0.0
        gain = max(0.0, self.last_distance - new_distance)
        return gain


def overlay_graph_relations(hyp: TCSHypothesis) -> None:
    # Example overlay: a simple proximity graph between objects
    try:
        import networkx as nx
    except Exception:
        return
    state = hyp.state
    graph = nx.Graph()
    objects = list(state.get_objects())
    for obj in objects:
        graph.add_node(obj)
    for i, a in enumerate(objects):
        pa = state.get_position(a)
        if pa is None:
            continue
        for b in objects[i + 1:]:
            pb = state.get_position(b)
            if pb is None:
                continue
            dist = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
            if dist <= 2:
                graph.add_edge(a, b, weight=1.0 / (1 + dist))
    hyp.add_overlay("graph", graph)


def overlay_call_stack(hyp: TCSHypothesis) -> None:
    hyp.add_overlay("call_stack", [
        {"frame": 0, "context": "root"},
    ]) 
