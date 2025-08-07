from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import heapq

from .tcs import TCSHypothesis
from .dsl import State
from .utils import cosine_similarity


@dataclass(order=True)
class SearchNode:
    priority: float
    neg_info_gain_per_cost: float = field(compare=False)
    depth: int = field(compare=False)
    hypothesis: TCSHypothesis = field(compare=False)
    operator_name: Optional[str] = field(default=None, compare=False)
    params: Optional[Dict[str, Any]] = field(default=None, compare=False)


@dataclass
class MCSController:
    max_expansions: int = 200
    fork_threshold: float = 0.05
    stall_patience: int = 30

    def search(self, hypotheses: List[TCSHypothesis], target: State, armory) -> Optional[TCSHypothesis]:
        # Initialize distances
        for hyp in hypotheses:
            hyp.set_progress(hyp.state.distance(target))

        queue: List[SearchNode] = []
        stall_counter = 0
        best_solution: Optional[TCSHypothesis] = None
        best_distance = float("inf")

        # Seed
        for hyp in hypotheses:
            self._enqueue_actions(queue, hyp, target, armory)

        expansions = 0
        while queue and expansions < self.max_expansions:
            node = heapq.heappop(queue)
            hyp = node.hypothesis

            # Apply operator
            if node.operator_name is not None and node.params is not None:
                op = armory.operators[node.operator_name]
                before_distance = hyp.state.distance(target)
                new_state = op.apply_kernel(hyp.state, node.params)
                new_distance = new_state.distance(target)
                info_gain = max(0.0, before_distance - new_distance)
                hyp.state = new_state
                hyp.record_action(node.operator_name, node.params, info_gain, op.cost_baseline)
                hyp.set_progress(new_distance)

            # Check improvement
            current_distance = hyp.last_distance if hyp.last_distance is not None else float("inf")
            if current_distance < best_distance:
                best_distance = current_distance
                best_solution = hyp
                stall_counter = 0
            else:
                stall_counter += 1

            # Goal check
            if current_distance == 0.0:
                return hyp

            # Escalation check
            if stall_counter > self.stall_patience:
                # Attempt analogy
                analog = armory.analogical_operator_synthesis(hyp.state, target)
                if analog is not None:
                    armory.register(analog)
                else:
                    # Conceptual void: memorize the current best mapping if known. Here we cannot auto-create after-state.
                    # We skip direct after-state; in a real system you would cache failed pairs for later deconstruction.
                    pass
                stall_counter = 0

            # Fork when two top operators are close
            top_two = armory.applicable(hyp.state)[:2]
            if len(top_two) >= 2 and abs(top_two[0][1] - top_two[1][1]) <= self.fork_threshold:
                # Fork hyp twice and enqueue both branches
                for i, (op, _) in enumerate(top_two[:2]):
                    forked = hyp.fork(tag=f"fork{i}")
                    self._enqueue_actions(queue, forked, target, armory, force_operator=op.name)
            else:
                # Enqueue next actions from current hyp
                self._enqueue_actions(queue, hyp, target, armory)

            expansions += 1

        return best_solution

    def _enqueue_actions(self, queue: List[SearchNode], hyp: TCSHypothesis, target: State, armory, force_operator: Optional[str] = None) -> None:
        applicable = armory.applicable(hyp.state)
        if force_operator is not None:
            applicable = [(armory.operators[force_operator], 1.0)]
        for op, score in applicable[:5]:  # limit branching factor
            # Predict ΔI via ghost
            proposals = op.propose_params(hyp.state, target)[:4]
            for params in proposals:
                ghost_fp = op.predict_effect(hyp.state, params)
                target_fp = compute_target_fingerprint(target)
                predicted_gain = cosine_similarity(target_fp, ghost_fp)
                cost = max(1e-6, float(op.cost_baseline))
                info_per_cost = float(predicted_gain) / cost
                node = SearchNode(
                    priority=-info_per_cost,  # min-heap
                    neg_info_gain_per_cost=-info_per_cost,
                    depth=len(hyp.history) + 1,
                    hypothesis=hyp,
                    operator_name=op.name,
                    params=params,
                )
                heapq.heappush(queue, node)


def compute_target_fingerprint(target: State):
    from .dsl import compute_structural_fingerprint
    return compute_structural_fingerprint(target) 
