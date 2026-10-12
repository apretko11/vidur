"""Ownership routing around the unchanged native resource executor.

Serving keys/callbacks remain stage/batch tuples. Aggregate keys are explicitly
tagged and never enter stage, request, token or serving-metric completion hooks.
"""

from dataclasses import asdict, dataclass, field
from math import isfinite

from vidur.entities.execution_plan import ExecutionPlan


@dataclass(frozen=True)
class AggregateOwner:
    work_id: str
    completion_id: str

    def __post_init__(self):
        if any(not isinstance(s, str) or not s.strip() for s in asdict(self).values()):
            raise ValueError(
                "aggregate ownership requires work and completion identities"
            )


@dataclass(frozen=True)
class ActivityTiming:
    activity_id: str
    start_seconds: float
    finish_seconds: float


@dataclass
class _AggregateState:
    plan: ExecutionPlan
    submitted_at: float
    submitted: bool = False
    observations: dict = field(default_factory=dict)
    completed_at: float | None = None


class NativeResourceContext:
    def __init__(self, executor, serving_context=None, context_id="aggregate"):
        self._resource_executor = executor
        self.serving_context = serving_context
        self.context_id = context_id
        self.aggregates = {}

    def reserve_aggregate(self, owner, plan, time):
        if not isinstance(owner, AggregateOwner):
            raise TypeError("aggregate submission requires explicit AggregateOwner")
        if type(time) not in (int, float) or not isfinite(time) or time < 0:
            raise ValueError("aggregate submission requires a finite nonnegative time")
        if any(
            o.work_id == owner.work_id or o.completion_id == owner.completion_id
            for o in self.aggregates
        ):
            raise ValueError("duplicate aggregate work or completion identity")
        if not isinstance(plan, ExecutionPlan) or type(plan.activities) is not tuple:
            raise ValueError(
                "aggregate submission requires an immutable execution plan"
            )
        plan.validate(self._resource_executor.capacities)
        if not plan.activities and not plan.limitations:
            raise ValueError("empty plans require an explicit omission boundary")
        for a in plan.activities:
            if a.memory_reads or a.memory_writes or a.memory_releases:
                raise NotImplementedError("aggregate-owned memory is not supported")
            if (
                type(a.duration_seconds) not in (int, float)
                or type(a.resources) is not tuple
                or type(a.dependencies) is not tuple
                or len(set(a.dependencies)) != len(a.dependencies)
            ):
                raise ValueError(
                    "aggregate resources/dependencies must be immutable and distinct"
                )
            if not a.cost_provenance.strip() or not a.cost_granularity.strip():
                raise ValueError("aggregate activities require evidenced cost bindings")
        self.aggregates[owner] = _AggregateState(plan, time)

    def submit_aggregate(self, owner, time):
        state = self.aggregates[owner]
        if state.submitted or time != state.submitted_at:
            raise ValueError("aggregate submission repeated or at an unexpected time")
        self._resource_executor.submit(owner, state.plan, time)
        state.submitted = True

    def on_activity_complete(self, key, activity, start, finish):
        if not isinstance(key, AggregateOwner):
            return
        state = self.aggregates[key]
        if activity.name in state.observations or state.completed_at is not None:
            raise RuntimeError("aggregate activity completed twice")
        state.observations[activity.name] = ActivityTiming(activity.name, start, finish)

    def on_native_plan_complete(self, key, time):
        if not isinstance(key, AggregateOwner):
            if self.serving_context is None:
                raise ValueError("serving completion requires a serving owner context")
            return self.serving_context.on_native_plan_complete(key, time)
        state = self.aggregates[key]
        if state.completed_at is not None:
            raise RuntimeError("aggregate completed twice")
        if not state.submitted or len(state.observations) != len(state.plan.activities):
            raise RuntimeError("aggregate completion requires every activity")
        state.completed_at = time
        return []

    def on_native_memory_release(self, time):
        if self.serving_context is None:
            raise NotImplementedError("aggregate-owned memory is not supported")
        return self.serving_context.on_native_memory_release(time)

    def aggregates_complete(self):
        return all(s.completed_at is not None for s in self.aggregates.values())

    def completion(self, owner):
        state = self.aggregates[owner]
        if state.completed_at is None:
            raise RuntimeError("aggregate execution is unfinished")
        elapsed = state.completed_at - state.submitted_at
        busy = dict.fromkeys(self._resource_executor.capacities, 0.0)
        for a in state.plan.activities:
            for resource in a.resources:
                busy[resource] += a.duration_seconds
        return {
            "work_kind": "aggregate",
            **asdict(owner),
            "submitted_at_seconds": state.submitted_at,
            "completed_at_seconds": state.completed_at,
            "makespan_seconds": elapsed,
            "total_service_seconds": sum(
                a.duration_seconds for a in state.plan.activities
            ),
            "resource_busy_seconds": busy,
            "resource_reservation_utilization": {
                r: seconds / (self._resource_executor.capacities[r] * elapsed)
                if elapsed
                else None
                for r, seconds in busy.items()
            },
            "activities": [
                asdict(state.observations[a.name]) for a in state.plan.activities
            ],
        }
