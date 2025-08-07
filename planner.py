from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .perception import PerceptionModel
from .tcs import TCSHypothesis
from .dsl import State
from .armory import Armory
from .mcs import MCSController
from .transcendental_core import TranscendentalCore


@dataclass
class Planner:
    perception: PerceptionModel
    armory: Armory
    mcs: MCSController
    tc: TranscendentalCore

    def solve(self, raw_input: Dict[str, Any], target_state: State) -> Optional[TCSHypothesis]:
        hyps = self.perception.perceive(raw_input)
        result = self.mcs.search(hyps, target_state, self.armory)
        if result is not None and result.last_distance == 0.0:
            # Log successful fingerprint
            self.armory.add_memory(result.state, {"operator": "solution", "type": "solved"})
            return result
        # If idle or partial, run curiosity cycles
        for _ in range(3):
            self.tc.curiosity_cycle(self.armory)
        self.tc.analyze_narrative()
        return result 
