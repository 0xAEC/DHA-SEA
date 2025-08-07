from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .operators import (
    SchemaOperator,
    move_operator,
    reflect_operator,
    rotate_operator,
    duplicate_shift_operator,
    recolor_operator,
    BlackBoxOperator,
)
from .dsl import State, compute_structural_fingerprint
from .utils import cosine_similarity


@dataclass
class Armory:
    operators: Dict[str, SchemaOperator] = field(default_factory=dict)
    black_boxes: List[BlackBoxOperator] = field(default_factory=list)
    fingerprint_memory: List[Tuple[np.ndarray, Dict[str, Any]]] = field(default_factory=list)

    @staticmethod
    def create_default() -> "Armory":
        arm = Armory()
        for op in [move_operator(), reflect_operator(), rotate_operator(), duplicate_shift_operator(), recolor_operator()]:
            arm.operators[op.name] = op
        return arm

    def register(self, op: SchemaOperator) -> None:
        self.operators[op.name] = op

    def add_memory(self, state: State, meta: Dict[str, Any]) -> None:
        self.fingerprint_memory.append((compute_structural_fingerprint(state), dict(meta)))

    def applicable(self, state: State) -> List[Tuple[SchemaOperator, float]]:
        scores: List[Tuple[SchemaOperator, float]] = []
        for op in self.operators.values():
            s = op.recognize(state)
            scores.append((op, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # -------- Analogy Engine --------

    def find_similar(self, state: State, k: int = 3) -> List[Dict[str, Any]]:
        fp = compute_structural_fingerprint(state)
        sims: List[Tuple[float, Dict[str, Any]]] = []
        for past_fp, meta in self.fingerprint_memory:
            sims.append((cosine_similarity(fp, past_fp), meta))
        sims.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in sims[:k]]

    def analogical_operator_synthesis(self, current_state: State, target: Optional[State]) -> Optional[SchemaOperator]:
        candidates = self.find_similar(current_state, k=5)
        if not candidates:
            return None
        # Localized heuristic matching: look for a common object and propose a move-like transform
        sample = candidates[0]
        if sample.get("operator") == "move" and target is not None:
            # Synthesize a specialized move-to-target operator
            def kernel(state: State, params: Dict[str, Any]) -> State:
                return self.operators["move"].apply_kernel(state, params)

            def propose(state: State, tgt: Optional[State]) -> List[Dict[str, Any]]:
                return self.operators["move"].propose_params(state, tgt)

            return SchemaOperator(
                name="analogical_move",
                recognizer=self.operators["move"].recognizer,
                apply_kernel=kernel,
                propose_params=propose,
                cost_baseline=0.9,
            )
        return None

    # -------- Conceptual Void Handling --------

    def declare_conceptual_void(self, before: State, after: State) -> BlackBoxOperator:
        bb = BlackBoxOperator(
            name=f"OP_Lambda_{len(self.black_boxes) + 1}",
            recognizer=self.operators["move"].recognizer,  # reuse a generic recognizer
            apply_kernel=lambda s, p: after,
            propose_params=lambda s, t: [{}],
            cost_baseline=2.0,
        )
        bb.remember(before, after)
        self.black_boxes.append(bb)
        self.register(bb)
        return bb 
