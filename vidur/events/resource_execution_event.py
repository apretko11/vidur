"""Native resource completion and dispatch events on Vidur's event heap."""

from vidur.entities import BatchStage
from vidur.events.base_event import BaseEvent
from vidur.scheduler.native_resource_context import AggregateOwner
from vidur.types import EventType


def submit_native_batch(time, replica_id, stage_id, stage):
    """Start the stage's next queued batch if the stage is idle.

    A stage runs one batch's plan at a time, as Vidur's slot path and real
    pipeline-parallel serving do; batches overlap only across stages. A plan
    may still run its own activities concurrently on different resources.
    """
    if stage._is_busy or not stage._batch_queue:
        return []
    from vidur.scheduler.execution_plan_builder import SelectedWork

    batch = stage._batch_queue.pop(0)
    plan = stage._execution_time_predictor.execution_plan_builder.build(
        SelectedWork.from_batch(batch, stage_id)
    )
    stage.resource_executor.submit((stage_id, batch.id), plan, time)
    stage._is_busy = True
    batch_stage = BatchStage(
        batch.id, replica_id, stage_id, None, None, batch.requests, batch.num_tokens
    )
    batch_stage.on_schedule(time)
    stage.native_batches[batch.id] = (batch, batch_stage)
    return [ResourceDispatchEvent(time, replica_id)]


class ResourceDispatchEvent(BaseEvent):
    def __init__(self, time, replica_id, *, context=None):
        super().__init__(time, EventType.RESOURCE_DISPATCH)
        self.replica_id = replica_id
        self.context = context

    def handle_event(self, scheduler, metrics_store):
        replica = self.context
        if replica is None:
            serving = scheduler.get_replica_scheduler(self.replica_id)
            replica = getattr(serving, "native_resource_context", None) or serving
        executor = replica._resource_executor
        events = []
        for key in list(executor.plans):
            if not executor.finished(key):
                continue
            executor.retire(key)
            events.extend(replica.on_native_plan_complete(key, self.time))
        for key, activity, start, finish in executor.dispatch(self.time):
            events.append(
                ResourceActivityEndEvent(
                    finish, self.replica_id, key, activity, start, context=self.context
                )
            )
        return events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": int(self._event_type),
            **(
                {"resource_context_id": self.context.context_id}
                if self.replica_id is None
                else {"replica_id": self.replica_id}
            ),
        }


class ResourceActivityEndEvent(BaseEvent):
    def __init__(self, time, replica_id, key, activity, start, *, context=None):
        super().__init__(time, EventType.RESOURCE_ACTIVITY_END)
        self.replica_id = replica_id
        self.key = key
        self.activity = activity
        self.start = start
        self.context = context

    def handle_event(self, scheduler, metrics_store):
        replica = self.context
        if replica is None:
            serving = scheduler.get_replica_scheduler(self.replica_id)
            replica = getattr(serving, "native_resource_context", None) or serving
        executor = replica._resource_executor
        executor.complete(self.key, self.activity.name, self.time)
        if isinstance(self.key, AggregateOwner):
            replica.on_activity_complete(self.key, self.activity, self.start, self.time)
        events = [
            ResourceDispatchEvent(self.time, self.replica_id, context=self.context)
        ]
        if self.activity.memory_releases:
            events.extend(replica.on_native_memory_release(self.time))
        return events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": int(self._event_type),
            **(
                {"resource_context_id": self.context.context_id}
                if self.replica_id is None
                else {"replica_id": self.replica_id}
            ),
            **(
                {
                    "work_kind": "aggregate",
                    "work_id": self.key.work_id,
                    "completion_id": self.key.completion_id,
                }
                if isinstance(self.key, AggregateOwner)
                else {"stage_id": self.key[0], "batch_id": self.key[1]}
            ),
            "activity": self.activity.name,
            "start_seconds": self.start,
            "duration_seconds": self.activity.duration_seconds,
            "resources": list(self.activity.resources),
            "dependencies": list(self.activity.dependencies),
            "cost_provenance": self.activity.cost_provenance,
            "cost_granularity": self.activity.cost_granularity,
            "memory_reads": [vars(access) for access in self.activity.memory_reads],
            "memory_writes": [vars(access) for access in self.activity.memory_writes],
            "memory_releases": [
                vars(access) for access in self.activity.memory_releases
            ],
            **({"plan_id": self.activity.plan_id} if self.activity.plan_id else {}),
        }

    def to_chrome_trace(self):
        return {
            "name": self.activity.name,
            "ph": "X",
            "pid": self.replica_id
            if self.replica_id is not None
            else self.context.context_id,
            "tid": "+".join(self.activity.resources),
            "ts": self.start * 1e6,
            "dur": (self.time - self.start) * 1e6,
            "args": self.to_dict(),
        }
