from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .dsl import State, Fact
from .tcs import TCSHypothesis
from .tcs import overlay_graph_relations, overlay_call_stack


@dataclass
class PerceptionModel:
    def perceive(self, raw_input: Dict[str, Any]) -> List[TCSHypothesis]:
        """Return ranked TCS hypotheses from raw input.

        raw_input format for the toy domain:
        {
          "grid_w": int, "grid_h": int,
          "objects": [
             {"id": str, "color": str, "x": int, "y": int}
          ]
        }
        """
        grid_w = int(raw_input.get("grid_w", 8))
        grid_h = int(raw_input.get("grid_h", 8))
        objects = list(raw_input.get("objects", []))

        # Hypothesis 1: As-is
        state1 = State(attributes={"grid_w": grid_w, "grid_h": grid_h})
        for o in objects:
            state1.facts.add(Fact("is_object", (o["id"],)))
            state1.facts.add(Fact("has_color", (o["id"], o.get("color", "red"))))
            state1.facts.add(Fact("has_position", (o["id"], int(o.get("x", 0)), int(o.get("y", 0)))))
        hyp1 = TCSHypothesis.create(state1, confidence=0.92)
        overlay_graph_relations(hyp1)
        overlay_call_stack(hyp1)

        # Hypothesis 2: Merge all into a composite object (toy interpretation)
        state2 = State(attributes={"grid_w": grid_w, "grid_h": grid_h})
        composite_id = "composite"
        state2.facts.add(Fact("is_object", (composite_id,)))
        state2.facts.add(Fact("has_color", (composite_id, "blue")))
        state2.facts.add(Fact("has_position", (composite_id, grid_w // 2, grid_h // 2)))
        hyp2 = TCSHypothesis.create(state2, confidence=0.61)

        # Hypothesis 3: Partition into foreground/background by y
        state3 = State(attributes={"grid_w": grid_w, "grid_h": grid_h})
        for o in objects:
            oid = f"fg_{o['id']}" if int(o.get("y", 0)) <= grid_h // 2 else f"bg_{o['id']}"
            state3.facts.add(Fact("is_object", (oid,)))
            state3.facts.add(Fact("has_color", (oid, o.get("color", "red"))))
            state3.facts.add(Fact("has_position", (oid, int(o.get("x", 0)), int(o.get("y", 0)))))
        hyp3 = TCSHypothesis.create(state3, confidence=0.55)

        return [hyp1, hyp2, hyp3] 
