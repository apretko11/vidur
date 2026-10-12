"""Declarative analytical plan construction, separate from admission and dispatch.

Policies receive immutable selected work, never live requests or schedulers.
They declare placement and structure; bindings supply seconds and evidence.
Only ResourceExecutor and the event loop decide start/finish times.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from vidur.entities.execution_plan import (
    ExecutionActivity,
    ExecutionPlan,
    MemoryAccess,
)


@dataclass(frozen=True)
class RequestState:
    request_id: int
    num_processed_tokens: int
    num_prefill_tokens: int
    is_prefill_complete: bool

    @classmethod
    def from_request(cls, request):
        return cls(
            request.id,
            request.num_processed_tokens,
            request.num_prefill_tokens,
            request.is_prefill_complete,
        )


@dataclass(frozen=True)
class SelectedRequest(RequestState):
    next_num_tokens: int


@dataclass(frozen=True)
class SelectedWork:
    requests: tuple[SelectedRequest, ...]
    pipeline_stage: int

    @classmethod
    def from_batch(cls, batch, pipeline_stage):
        if len(batch.requests) != len(batch.num_tokens):
            raise ValueError("selected requests and token counts must align")
        return cls(
            tuple(
                SelectedRequest(
                    r.id,
                    r.num_processed_tokens,
                    r.num_prefill_tokens,
                    r.is_prefill_complete,
                    count,
                )
                for r, count in zip(batch.requests, batch.num_tokens)
            ),
            pipeline_stage,
        )

    def __post_init__(self):
        if type(self.pipeline_stage) is not int or self.pipeline_stage < 0:
            raise ValueError("pipeline stage must be a nonnegative integer")
        if type(self.requests) is not tuple or any(
            not isinstance(r, SelectedRequest) for r in self.requests
        ):
            raise ValueError(
                "selected requests must be an immutable tuple of snapshots"
            )
        ids = [r.request_id for r in self.requests]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("selected work requires distinct requests")
        for r in self.requests:
            if (
                any(
                    type(n) is not int or n < 0
                    for n in (
                        r.request_id,
                        r.num_processed_tokens,
                        r.num_prefill_tokens,
                    )
                )
                or type(r.next_num_tokens) is not int
                or r.next_num_tokens < 1
            ):
                raise ValueError("invalid selected token counts")
            if type(r.is_prefill_complete) is not bool:
                raise ValueError("prefill completion must be a boolean")
            if r.is_prefill_complete:
                if r.num_processed_tokens <= r.num_prefill_tokens:
                    raise ValueError(
                        "completed prefill must include the first output token"
                    )
            elif (
                r.num_processed_tokens >= r.num_prefill_tokens
                or r.num_processed_tokens + r.next_num_tokens > r.num_prefill_tokens
            ):
                raise ValueError("prefill selection exceeds remaining prompt tokens")
            if r.is_prefill_complete and r.next_num_tokens != 1:
                raise ValueError("decode selection requires one token per request")


@dataclass(frozen=True)
class ActivityDeclaration:
    name: str
    cost_key: str
    resources: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    memory_reads: tuple[MemoryAccess, ...] = ()
    memory_writes: tuple[MemoryAccess, ...] = ()
    memory_releases: tuple[MemoryAccess, ...] = ()


@dataclass(frozen=True)
class PlanDeclaration:
    activities: tuple[ActivityDeclaration, ...]
    # Empty/omitted work needs an explicit evidence boundary too.
    limitations: tuple[str, ...] = ()
    # Optional identity copied onto every activity and its trace events.
    plan_id: str = ""


@dataclass(frozen=True)
class BoundCost:
    seconds: float
    provenance: str
    granularity: str

    def __post_init__(self):
        if not isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("bound costs must be finite nonnegative seconds")
        if not self.provenance.strip() or not self.granularity.strip():
            raise ValueError("bound costs require provenance and granularity")


class ArchitecturePolicy(Protocol):
    def declare(self, work: SelectedWork) -> PlanDeclaration: ...

    def resource_capacities(self) -> dict[str, int]: ...

    def memory_pool_capacities(self) -> dict[str, int]: ...


class CostBinding(Protocol):
    def bind(self, work: SelectedWork) -> dict[str, BoundCost]:
        """Validate the calibration domain and return evidenced costs by key."""
        ...


class MemoryPolicy(Protocol):
    """Demand only; allocation, readiness and eviction belong to Vidur.

    A None step demand retains the fixed reservation contract. No realized
    future output length is exposed by RequestState.
    """

    def request_memory_requirements(self, request: RequestState) -> dict[str, int]: ...

    def request_step_memory_requirements(
        self, request: RequestState, next_num_tokens: int
    ) -> dict[str, int] | None: ...


class ExecutionPlanBuilder:
    def __init__(self, policy: ArchitecturePolicy, costs: CostBinding):
        self.policy = policy
        self.costs = costs

    def build(self, work: SelectedWork) -> ExecutionPlan:
        declaration = self.policy.declare(work)
        costs = self.costs.bind(work)
        capacities = self.policy.resource_capacities()
        pools = self.policy.memory_pool_capacities()
        ids = {r.request_id for r in work.requests}

        def validate_access(access):
            if access.request_id not in ids or access.pool not in pools:
                raise ValueError(
                    "memory access outside selected work or declared pools"
                )

        return self.build_declaration(declaration, costs, capacities, validate_access)

    @staticmethod
    def build_declaration(declaration, costs, capacities, access_validator=None):
        """Materialize evidenced work without requiring serving snapshots.

        Without an explicit memory-owner validator, accesses fail closed. The
        serving entry supplies exactly its existing selected-request checks.
        This method never selects work, schedules activities or advances tokens.
        """
        if not capacities or any(
            not isinstance(name, str) or not name or type(n) is not int or n < 1
            for name, n in capacities.items()
        ):
            raise ValueError("resource capacities must be named positive integers")
        activities = []
        for activity in declaration.activities:
            if activity.cost_key not in costs:
                raise ValueError(f"missing cost binding: {activity.cost_key}")
            for access in (
                activity.memory_reads
                + activity.memory_writes
                + activity.memory_releases
            ):
                if access_validator is None:
                    raise ValueError(
                        "memory accesses require an explicit owner validator"
                    )
                access_validator(access)
            cost = costs[activity.cost_key]
            if not isinstance(cost, BoundCost):
                raise TypeError("cost bindings require evidenced BoundCost values")
            activities.append(
                ExecutionActivity(
                    activity.name,
                    cost.seconds,
                    activity.resources,
                    activity.dependencies,
                    activity.memory_reads,
                    activity.memory_writes,
                    activity.memory_releases,
                    cost.provenance,
                    cost.granularity,
                    declaration.plan_id,
                )
            )
        if not activities and not declaration.limitations:
            raise ValueError("empty plans require an explicit omission boundary")
        plan = ExecutionPlan(tuple(activities), declaration.limitations)
        plan.validate(capacities)
        return plan
