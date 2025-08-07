from __future__ import annotations

from typing import Any, Dict

from unified_architecture.dsl import State, Fact


def sample_raw_and_target() -> (Dict[str, Any], State):
    raw = {
        "grid_w": 8,
        "grid_h": 8,
        "objects": [
            {"id": "a", "color": "red", "x": 1, "y": 1},
            {"id": "b", "color": "green", "x": 6, "y": 6},
        ],
    }

    target = State(attributes={"grid_w": 8, "grid_h": 8})
    target.facts.add(Fact("is_object", ("a",)))
    target.facts.add(Fact("has_color", ("a", "red")))
    target.facts.add(Fact("has_position", ("a", 4, 4)))

    target.facts.add(Fact("is_object", ("b",)))
    target.facts.add(Fact("has_color", ("b", "green")))
    target.facts.add(Fact("has_position", ("b", 4, 4)))

    return raw, target 
