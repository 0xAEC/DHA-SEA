from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .dsl import State
from .operators import SchemaOperator


@dataclass
class ProgramSynthesisLearner:
    templates: List[str] = None

    def __post_init__(self) -> None:
        if self.templates is None:
            self.templates = ["move", "reflect", "rotate", "duplicate_and_shift", "recolor"]

    def synthesize(self, demos: List[Tuple[State, State]], armory) -> Optional[SchemaOperator]:
        # Try to fit known operator templates by checking deltas
        for template in self.templates:
            op = armory.operators.get(template)
            if op is None:
                continue
            if self._fits(op, demos):
                return op
        return None

    def _fits(self, op: SchemaOperator, demos: List[Tuple[State, State]]) -> bool:
        # Simple heuristic: operator must be able to transform input to output for some params proposal
        for before, after in demos:
            proposals = op.propose_params(before, after)
            found = False
            for params in proposals:
                result = op.apply_kernel(before, params)
                if result.distance(after) == 0.0:
                    found = True
                    break
            if not found:
                return False
        return True 
