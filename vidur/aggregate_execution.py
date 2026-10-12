"""Bounded, memory-free aggregate execution on Vidur's native event heap.

No Cluster, model, requests, batches, token progress or serving MetricsStore is
constructed. This entry reuses the exact loop and resource events used by serving.
"""

from dataclasses import dataclass
from math import isfinite

from vidur.entities.execution_plan import ExecutionPlan
from vidur.event_loop import EventLoop
from vidur.events.base_event import BaseEvent
from vidur.events.resource_execution_event import ResourceDispatchEvent
from vidur.scheduler.native_resource_context import (
    AggregateOwner,
    NativeResourceContext,
)
from vidur.scheduler.resource_executor import ResourceExecutor
from vidur.types import EventType


@dataclass(frozen=True)
class AggregateSubmission:
    owner: AggregateOwner
    plan: ExecutionPlan
    submitted_at_seconds: float = 0.0


class AggregateSubmitEvent(BaseEvent):
    def __init__(self, submission, context, replica_id=None):
        super().__init__(submission.submitted_at_seconds, EventType.AGGREGATE_SUBMIT)
        self.submission, self.context, self.replica_id = submission, context, replica_id

    def handle_event(self, scheduler, metrics_store):
        self.context.submit_aggregate(self.submission.owner, self.time)
        return [ResourceDispatchEvent(self.time, self.replica_id, context=self.context)]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": int(self._event_type),
            "work_kind": "aggregate",
            "work_id": self.submission.owner.work_id,
            "completion_id": self.submission.owner.completion_id,
            "resource_context_id": self.context.context_id,
        }


def enqueue_aggregate(loop, context, submission, replica_id=None):
    """Validate ownership before queuing one submission on the existing heap."""
    if not isinstance(submission, AggregateSubmission):
        raise TypeError("aggregate execution requires AggregateSubmission")
    time = submission.submitted_at_seconds
    if type(time) not in (int, float) or not isfinite(time) or time < loop._time:
        raise ValueError("events must have finite, nondecreasing timestamps")
    context.reserve_aggregate(submission.owner, submission.plan, time)
    loop._add_event(AggregateSubmitEvent(submission, context, replica_id))


class AggregateExecutionSimulation(EventLoop):
    def __init__(self, capacities, submissions, *, time_limit=0):
        if type(submissions) is not tuple or not submissions:
            raise ValueError(
                "aggregate program requires nonempty immutable submissions"
            )
        if (
            type(time_limit) not in (int, float)
            or not isfinite(time_limit)
            or time_limit < 0
        ):
            raise ValueError(
                "time limit must be finite nonnegative seconds (0 means unlimited)"
            )
        super().__init__(time_limit, write_json_trace=True, enable_chrome_trace=True)
        self.context = NativeResourceContext(ResourceExecutor(capacities))
        # EventLoop requires only is_empty; aggregate events carry their explicit
        # context and never look up a replica or serving scheduler.
        self._scheduler = self
        self._metric_store = None
        for submission in submissions:
            enqueue_aggregate(self, self.context, submission)

    def is_empty(self):
        executor = self.context._resource_executor
        return (
            self.context.aggregates_complete()
            and not executor.plans
            and not executor.running
        )

    def diagnostics(self):
        return {
            "current_time_seconds": self._time,
            "terminated_at_time_limit": self._terminate,
            "resource_used": dict(self.context._resource_executor.used),
            "events": list(self._event_trace),
            "traceEvents": list(self._event_chrome_trace),
        }

    def report(self):
        if not self.is_empty():
            raise RuntimeError("aggregate execution is unfinished")
        return {
            "schema": "vidur.aggregate-execution.v1",
            "unsupported_metrics": [
                "TTFT",
                "TPOT",
                "request latency",
                "token throughput",
            ],
            "completions": [
                self.context.completion(owner) for owner in self.context.aggregates
            ],
            **self.diagnostics(),
        }
