from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np

from .dsl import State, DSL, compute_structural_fingerprint


@dataclass
class Recognizer:
    name: str
    weight: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=float))
    bias: float = 0.0

    def score(self, fingerprint: np.ndarray) -> float:
        w = self._ensure_shape(fingerprint)
        z = float(np.dot(w, fingerprint) + self.bias)
        return 1.0 / (1.0 + np.exp(-z))

    def train(self, examples: List[Tuple[np.ndarray, int]], lr: float = 0.1, epochs: int = 3) -> None:
        for _ in range(epochs):
            for fp, label in examples:
                w = self._ensure_shape(fp)
                pred = 1.0 / (1.0 + np.exp(-(float(np.dot(w, fp)) + self.bias)))
                error = float(label) - pred
                self.weight = w + lr * error * fp
                self.bias += lr * error

    def _ensure_shape(self, fp: np.ndarray) -> np.ndarray:
        if self.weight.shape != fp.shape:
            # Initialize to small random values to match fp shape
            rng = np.random.default_rng(42)
            self.weight = rng.normal(loc=0.0, scale=0.05, size=fp.shape)
        return self.weight


@dataclass
class EffectPredictor:
    name: str

    def predict_fingerprint(self, state: State, simulate: Callable[[State], State]) -> np.ndarray:
        ghost_state = simulate(state)
        return compute_structural_fingerprint(ghost_state)


@dataclass
class SchemaOperator:
    name: str
    recognizer: Recognizer
    apply_kernel: Callable[[State, Dict[str, Any]], State]
    propose_params: Callable[[State, Optional[State]], List[Dict[str, Any]]]
    effect_predictor: EffectPredictor = field(default_factory=lambda: EffectPredictor(name="ghost"))
    cost_baseline: float = 1.0

    def recognize(self, state: State) -> float:
        fp = compute_structural_fingerprint(state)
        return self.recognizer.score(fp)

    def predict_effect(self, state: State, params: Dict[str, Any]) -> np.ndarray:
        return self.effect_predictor.predict_fingerprint(state, lambda s: self.apply_kernel(s, params))


# -------------------- Built-in Operators --------------------


def move_operator() -> SchemaOperator:
    def kernel(state: State, params: Dict[str, Any]) -> State:
        return DSL.move(state, params["obj_id"], int(params.get("dx", 0)), int(params.get("dy", 0)))

    def propose(state: State, target: Optional[State]) -> List[Dict[str, Any]]:
        params: List[Dict[str, Any]] = []
        for obj in state.get_objects():
            # Propose small moves in 4-neighborhood; if target known, bias towards it
            if target is not None:
                p = state.get_position(obj)
                q = target.get_position(obj)
                if p is not None and q is not None:
                    dx = np.sign(q[0] - p[0])
                    dy = np.sign(q[1] - p[1])
                    params.append({"obj_id": obj, "dx": int(dx), "dy": int(dy)})
            params.extend([
                {"obj_id": obj, "dx": 1, "dy": 0},
                {"obj_id": obj, "dx": -1, "dy": 0},
                {"obj_id": obj, "dx": 0, "dy": 1},
                {"obj_id": obj, "dx": 0, "dy": -1},
            ])
        return params

    return SchemaOperator(
        name="move",
        recognizer=Recognizer(name="move_recognizer"),
        apply_kernel=kernel,
        propose_params=propose,
        cost_baseline=1.0,
    )


def reflect_operator() -> SchemaOperator:
    def kernel(state: State, params: Dict[str, Any]) -> State:
        return DSL.reflect(state, params["obj_id"], params.get("axis", "x"))

    def propose(state: State, target: Optional[State]) -> List[Dict[str, Any]]:
        params: List[Dict[str, Any]] = []
        for obj in state.get_objects():
            for axis in ["x", "y", "color"]:
                params.append({"obj_id": obj, "axis": axis})
        return params

    return SchemaOperator(
        name="reflect",
        recognizer=Recognizer(name="reflect_recognizer"),
        apply_kernel=kernel,
        propose_params=propose,
        cost_baseline=1.2,
    )


def rotate_operator() -> SchemaOperator:
    def kernel(state: State, params: Dict[str, Any]) -> State:
        return DSL.rotate(state, params["obj_id"], int(params.get("quarter_turns", 1)))

    def propose(state: State, target: Optional[State]) -> List[Dict[str, Any]]:
        params: List[Dict[str, Any]] = []
        for obj in state.get_objects():
            for qt in [1, 2, 3]:
                params.append({"obj_id": obj, "quarter_turns": qt})
        return params

    return SchemaOperator(
        name="rotate",
        recognizer=Recognizer(name="rotate_recognizer"),
        apply_kernel=kernel,
        propose_params=propose,
        cost_baseline=1.5,
    )


def duplicate_shift_operator() -> SchemaOperator:
    def kernel(state: State, params: Dict[str, Any]) -> State:
        return DSL.duplicate_and_shift(state, params["src_obj_id"], params["new_obj_id"], int(params.get("dx", 0)), int(params.get("dy", 0)))

    def propose(state: State, target: Optional[State]) -> List[Dict[str, Any]]:
        params: List[Dict[str, Any]] = []
        for obj in state.get_objects():
            new_id = f"{obj}_copy"
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                params.append({"src_obj_id": obj, "new_obj_id": new_id, "dx": dx, "dy": dy})
        return params

    return SchemaOperator(
        name="duplicate_and_shift",
        recognizer=Recognizer(name="dup_shift_recognizer"),
        apply_kernel=kernel,
        propose_params=propose,
        cost_baseline=1.8,
    )


def recolor_operator() -> SchemaOperator:
    def kernel(state: State, params: Dict[str, Any]) -> State:
        return DSL.recolor(state, params["obj_id"], params.get("color", "red"))

    def propose(state: State, target: Optional[State]) -> List[Dict[str, Any]]:
        palette = ["red", "green", "blue", "yellow", "cyan", "magenta", "black", "white"]
        params: List[Dict[str, Any]] = []
        for obj in state.get_objects():
            for c in palette:
                params.append({"obj_id": obj, "color": c})
        return params

    return SchemaOperator(
        name="recolor",
        recognizer=Recognizer(name="recolor_recognizer"),
        apply_kernel=kernel,
        propose_params=propose,
        cost_baseline=0.8,
    )


@dataclass
class BlackBoxOperator(SchemaOperator):
    """A black-box operator created during conceptual voids.

    It memorizes input-output pairs and can replay them when matching contexts
    are detected. Over time it can be refined by the meta-learner.
    """
    memory_pairs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def remember(self, before_state: State, after_state: State) -> None:
        self.memory_pairs.append(
            (compute_structural_fingerprint(before_state), compute_structural_fingerprint(after_state))
        ) 
