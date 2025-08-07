from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random

from .memory import NarrativeLog, FingerprintStore
from .dsl import State


@dataclass
class PrincipledHeuristic:
    name: str
    weight: float
    description: str


@dataclass
class TranscendentalCore:
    narrative: NarrativeLog = field(default_factory=NarrativeLog)
    fingerprints: FingerprintStore = field(default_factory=FingerprintStore)
    intrinsic_goals: List[str] = field(default_factory=lambda: ["Seek Elegance", "Master Contiguity"])
    heuristics: List[PrincipledHeuristic] = field(default_factory=lambda: [
        PrincipledHeuristic(name="Elegance", weight=0.3, description="Maximal effect from minimal action"),
        PrincipledHeuristic(name="Symmetry", weight=0.2, description="Prefer symmetric intermediate states"),
    ])

    def curiosity_cycle(self, armory) -> None:
        # Generate a hypothetical problem by chaining random operators and observe behavior
        ops = list(armory.operators.values())
        if not ops:
            return
        # Start from a minimal state
        state = State(attributes={"grid_w": 8, "grid_h": 8})
        obj = "o"
        state.facts.update({
            # seed object
            })
        state.facts.add(__import__("unified_architecture.dsl", fromlist=["Fact"]).Fact("is_object", (obj,)))
        state.facts.add(__import__("unified_architecture.dsl", fromlist=["Fact"]).Fact("has_color", (obj, "red")))
        state.facts.add(__import__("unified_architecture.dsl", fromlist=["Fact"]).Fact("has_position", (obj, 4, 4)))

        op = random.choice(ops)
        proposals = op.propose_params(state, None)
        if not proposals:
            return
        params = random.choice(proposals)
        new_state = op.apply_kernel(state, params)

        # Record into memory and narrative log
        self.fingerprints.add(new_state, {"operator": op.name, "type": "curiosity_outcome"})
        self.narrative.record({
            "type": "curiosity_cycle",
            "operator": op.name,
            "params": params,
        })

    def analyze_narrative(self) -> None:
        summary = self.narrative.summarize()
        # Emergent goal formulation: if certain event types dominate, bias goals
        if summary.get("conceptual_void", 0) > 3 and "Master Topology" not in self.intrinsic_goals:
            self.intrinsic_goals.append("Master Topology")

    def heuristic_bonus(self, path_properties: Dict[str, Any]) -> float:
        # Example: reward short paths and minimal changes
        bonus = 0.0
        length = int(path_properties.get("length", 0))
        delta_facts = float(path_properties.get("delta_facts", 0.0))
        if length <= 3:
            bonus += 0.2
        if delta_facts <= 5:
            bonus += 0.1
        for h in self.heuristics:
            bonus += 0.05 * h.weight
        return bonus 
