from __future__ import annotations

from unified_architecture.perception import PerceptionModel
from unified_architecture.armory import Armory
from unified_architecture.mcs import MCSController
from unified_architecture.transcendental_core import TranscendentalCore
from unified_architecture.planner import Planner
from examples.sample_problems import sample_raw_and_target


def main() -> None:
    raw, target = sample_raw_and_target()

    planner = Planner(
        perception=PerceptionModel(),
        armory=Armory.create_default(),
        mcs=MCSController(max_expansions=300, fork_threshold=0.02, stall_patience=40),
        tc=TranscendentalCore(),
    )

    result = planner.solve(raw, target)

    if result is None:
        print("No solution found. Partial results or curiosity cycles may have run.")
        return

    dist = result.last_distance
    print(f"Best distance to goal: {dist}")
    print("Action history:")
    for step, h in enumerate(result.history, 1):
        print(f"  {step:02d}. {h['operator']} {h['params']} (ΔI={h['ΔI']:.3f}, ΔC={h['ΔC']:.2f})")


if __name__ == "__main__":
    main() 
