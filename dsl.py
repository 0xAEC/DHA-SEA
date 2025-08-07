from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class Fact:
    predicate: str
    arguments: Tuple[Any, ...]

    def __str__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.arguments)
        return f"{self.predicate}({args_str})"


@dataclass
class State:
    """Symbolic state as a set of facts in the foundational DSL.

    Facts include:
      - is_object(ID)
      - has_color(ID, color)
      - has_position(ID, x, y)

    Attributes store metadata like grid_size or domain-specific parameters.
    """

    facts: Set[Fact] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "State":
        return State(facts=set(self.facts), attributes=dict(self.attributes))

    def add_fact(self, fact: Fact) -> None:
        self.facts.add(fact)

    def remove_fact(self, fact: Fact) -> None:
        self.facts.discard(fact)

    def replace_fact(self, old_fact: Fact, new_fact: Fact) -> None:
        if old_fact in self.facts:
            self.facts.discard(old_fact)
        self.facts.add(new_fact)

    # -------------------- DSL Query Helpers --------------------

    def get_objects(self) -> Set[str]:
        return {f.arguments[0] for f in self.facts if f.predicate == "is_object"}

    def get_color(self, obj_id: str) -> Optional[str]:
        for f in self.facts:
            if f.predicate == "has_color" and f.arguments[0] == obj_id:
                return str(f.arguments[1])
        return None

    def get_position(self, obj_id: str) -> Optional[Tuple[int, int]]:
        for f in self.facts:
            if f.predicate == "has_position" and f.arguments[0] == obj_id:
                return int(f.arguments[1]), int(f.arguments[2])
        return None

    def set_position(self, obj_id: str, pos: Tuple[int, int]) -> None:
        old = None
        for f in list(self.facts):
            if f.predicate == "has_position" and f.arguments[0] == obj_id:
                old = f
                break
        if old is not None:
            self.facts.discard(old)
        self.facts.add(Fact("has_position", (obj_id, int(pos[0]), int(pos[1]))))

    def set_color(self, obj_id: str, color: str) -> None:
        old = None
        for f in list(self.facts):
            if f.predicate == "has_color" and f.arguments[0] == obj_id:
                old = f
                break
        if old is not None:
            self.facts.discard(old)
        self.facts.add(Fact("has_color", (obj_id, color)))

    # -------------------- Serialization --------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": [
                {"predicate": f.predicate, "arguments": list(f.arguments)}
                for f in sorted(self.facts, key=str)
            ],
            "attributes": dict(self.attributes),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "State":
        st = State(
            facts={
                Fact(item["predicate"], tuple(item["arguments"]))
                for item in data.get("facts", [])
            },
            attributes=dict(data.get("attributes", {})),
        )
        return st

    # -------------------- Distance and Diff --------------------

    def diff(self, other: "State") -> Dict[str, Any]:
        added = other.facts - self.facts
        removed = self.facts - other.facts
        return {"added": added, "removed": removed}

    def distance(self, other: "State") -> float:
        # Position and color differences over shared objects
        objects = self.get_objects().union(other.get_objects())
        pos_dist, color_dist = 0.0, 0.0
        for obj in objects:
            p1, p2 = self.get_position(obj), other.get_position(obj)
            if p1 is None or p2 is None:
                pos_dist += 1.0
            else:
                pos_dist += manhattan_distance(p1, p2)
            c1, c2 = self.get_color(obj), other.get_color(obj)
            if c1 != c2:
                color_dist += 1.0
        # Fact-level symmetric difference penalty
        fact_symm = len(self.facts.symmetric_difference(other.facts)) * 0.1
        return pos_dist + color_dist + fact_symm


# -------------------- DSL Transformations --------------------

class DSL:
    @staticmethod
    def move(state: State, obj_id: str, delta_x: int, delta_y: int) -> State:
        new_state = state.copy()
        pos = new_state.get_position(obj_id)
        if pos is None:
            return new_state
        grid_w, grid_h = get_grid_size(new_state)
        nx = max(0, min(grid_w - 1, pos[0] + int(delta_x)))
        ny = max(0, min(grid_h - 1, pos[1] + int(delta_y)))
        new_state.set_position(obj_id, (nx, ny))
        return new_state

    @staticmethod
    def reflect(state: State, obj_id: str, axis: str) -> State:
        new_state = state.copy()
        pos = new_state.get_position(obj_id)
        if pos is None:
            return new_state
        grid_w, grid_h = get_grid_size(new_state)
        if axis == "x":
            new_state.set_position(obj_id, (grid_w - 1 - pos[0], pos[1]))
        elif axis == "y":
            new_state.set_position(obj_id, (pos[0], grid_h - 1 - pos[1]))
        elif axis == "color":
            color = new_state.get_color(obj_id)
            new_state.set_color(obj_id, reflect_color(color))
        return new_state

    @staticmethod
    def rotate(state: State, obj_id: str, quarter_turns: int) -> State:
        new_state = state.copy()
        qt = int(quarter_turns) % 4
        if qt == 0:
            return new_state
        pos = new_state.get_position(obj_id)
        if pos is None:
            return new_state
        grid_w, grid_h = get_grid_size(new_state)
        x, y = pos
        cx, cy = (grid_w - 1) / 2.0, (grid_h - 1) / 2.0
        # Translate, rotate around center, translate back
        rx, ry = x - cx, y - cy
        for _ in range(qt):
            rx, ry = -ry, rx
        nx, ny = int(round(rx + cx)), int(round(ry + cy))
        nx = max(0, min(grid_w - 1, nx))
        ny = max(0, min(grid_h - 1, ny))
        new_state.set_position(obj_id, (nx, ny))
        return new_state

    @staticmethod
    def duplicate_and_shift(state: State, src_obj_id: str, new_obj_id: str, delta_x: int, delta_y: int) -> State:
        new_state = state.copy()
        if src_obj_id not in new_state.get_objects():
            return new_state
        new_state.add_fact(Fact("is_object", (new_obj_id,)))
        color = new_state.get_color(src_obj_id)
        if color is not None:
            new_state.add_fact(Fact("has_color", (new_obj_id, color)))
        pos = new_state.get_position(src_obj_id)
        if pos is not None:
            grid_w, grid_h = get_grid_size(new_state)
            nx = max(0, min(grid_w - 1, pos[0] + int(delta_x)))
            ny = max(0, min(grid_h - 1, pos[1] + int(delta_y)))
            new_state.add_fact(Fact("has_position", (new_obj_id, nx, ny)))
        return new_state

    @staticmethod
    def recolor(state: State, obj_id: str, new_color: str) -> State:
        new_state = state.copy()
        new_state.set_color(obj_id, new_color)
        return new_state


# -------------------- Structural Fingerprint --------------------

def compute_structural_fingerprint(state: State, *, num_colors: int = 8) -> np.ndarray:
    grid_w, grid_h = get_grid_size(state)
    objects = sorted(state.get_objects())

    # Position stats
    xs, ys = [], []
    for obj in objects:
        pos = state.get_position(obj)
        if pos is not None:
            xs.append(pos[0] / max(1, grid_w - 1))
            ys.append(pos[1] / max(1, grid_h - 1))
    if len(xs) == 0:
        xs = [0.5]
        ys = [0.5]
    pos_stats = [
        float(np.mean(xs)), float(np.mean(ys)),
        float(np.std(xs)), float(np.std(ys)),
        float(np.min(xs)), float(np.max(xs)),
        float(np.min(ys)), float(np.max(ys)),
    ]

    # Color histogram (hash colors to bins)
    color_bins = np.zeros(num_colors, dtype=float)
    for obj in objects:
        color = state.get_color(obj)
        if color is None:
            continue
        bin_idx = hash_color_to_bin(color, num_colors)
        color_bins[bin_idx] += 1.0
    if len(objects) > 0:
        color_bins /= float(len(objects))

    # Object count
    count_feat = [float(len(objects))]

    return np.array(pos_stats + count_feat + color_bins.tolist(), dtype=float)


# -------------------- Utilities --------------------

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def get_grid_size(state: State) -> Tuple[int, int]:
    grid_w = int(state.attributes.get("grid_w", 8))
    grid_h = int(state.attributes.get("grid_h", 8))
    return grid_w, grid_h


def hash_color_to_bin(color: Optional[str], num_bins: int) -> int:
    if color is None:
        return 0
    return abs(hash(str(color))) % max(1, num_bins)


def reflect_color(color: Optional[str]) -> str:
    if color is None:
        return "unknown"
    # Simple color-space reflection mapping over a fixed palette
    palette = [
        "red", "green", "blue", "yellow", "cyan", "magenta", "black", "white",
    ]
    if color not in palette:
        return color
    idx = palette.index(color)
    return palette[len(palette) - 1 - idx] 
