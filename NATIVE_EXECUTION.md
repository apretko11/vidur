# Native analytical resource execution

This fork adds an opt-in execution path alongside Vidur's operator-slot
predictors. A registered predictor returns a `NativeExecutionProvider` that
wraps an `ExecutionPlanBuilder(policy, costs)` and an optional memory policy:

- `policy.declare(work)` turns immutable selected work (`SelectedWork`: request
  snapshots and the pipeline stage) into a `PlanDeclaration` of activities,
  each with resources, dependencies and memory accesses, plus limitations and
  an optional plan identity.
- `costs.bind(work)` returns seconds with provenance and granularity per cost
  key, and raises outside its calibration domain.
- `policy.resource_capacities()` and `memory_pool_capacities()` name the
  replica's resources and byte pools.

Activity completions and resource dispatch run on the normal event heap. A
stage runs one batch's plan at a time, as the slot path does; batches overlap
across stages. A replica-wide executor starts dependency-ready activities in
plan order, atomically reserves one unit of every named resource, and releases
it on completion. It does not reserve resources for future work or infer
interference costs; costs are fixed once a plan is submitted. This is an
analytical model, not hardware execution or cycle simulation.

Memory pools admit requests by declared byte demand (a fixed reservation or
step growth) and track resident prefixes: an activity starts only when its
reads are resident, and its writes become resident on completion. A request
that arrives with tokens already processed (a warm start) is seeded with its
resident prefix.

Native vLLM and Sarathi configurations can decouple `max_inflight_batches` and
`micro_batch_size` from pipeline depth. All activities in a stage complete
before stage and batch events advance tokens and release KV.

Chrome traces include activity intervals, resources, dependencies, declared
duration, plan identity, and stage envelopes with per-request prefill status.
GPU MFU and operator-slot attribution are omitted on this path. The heap orders
events by type before ID, so completions run before dispatch at equal times. A
simulation that stops with unfinished requests raises instead of asserting.

Run core tests with `python -m unittest discover -s tests`. PIMScope carries
end-to-end serving, request-lifecycle, pipeline and calibrated-model regressions.

## Request-free aggregate work (opt-in)

`ExecutionPlanBuilder.build_declaration(declaration, costs, capacities)` performs
the same pure materialization without SelectedWork. Accesses require an explicit
owner validator; ordinary `build(work)` supplies its original request/pool checks.
Aggregate execution currently rejects all memory accesses, including on a
replica with serving memory pools. No aggregate allocation semantics are inferred.

```python
from vidur.aggregate_execution import AggregateExecutionSimulation, AggregateSubmission
from vidur.scheduler.execution_plan_builder import (
    ActivityDeclaration, BoundCost, ExecutionPlanBuilder, PlanDeclaration,
)
from vidur.scheduler.native_resource_context import AggregateOwner

declaration = PlanDeclaration((
    ActivityDeclaration("A", "A", ("R0",)),
    ActivityDeclaration("B", "B", ("R1",)),
    ActivityDeclaration("C", "C", ("R0",), ("A", "B")),
    ActivityDeclaration("D", "D", ("R0",)),
), plan_id="synthetic-resource-fixture")
costs = {name: BoundCost(seconds, "Synthetic seconds; no calibration", "Activity")
         for name, seconds in {"A": 3, "B": 2, "C": 1, "D": 2}.items()}
capacities = {"R0": 1, "R1": 1}
plan = ExecutionPlanBuilder.build_declaration(declaration, costs, capacities)
simulation = AggregateExecutionSimulation(capacities, (
    AggregateSubmission(AggregateOwner("slice_0", "slice_complete"), plan),
))
simulation.run()
result = simulation.report()
assert result["completions"][0]["makespan_seconds"] == 6
```

Increasing R0 capacity to two gives four seconds. These are correctness
fixtures, not hardware measurements. All owners in a program share one unchanged
ResourceExecutor. Multiple bounded submissions have explicit ready times and
FIFO/activity tie breaking. The public Simulator.submit_aggregate(replica_id,
submission) also permits resource-only aggregate work to contend with genuine
serving plans in that native replica. Slot-only replicas reject it.

The earlier c793dd8 EventLoop extraction is reused by both Simulator and
AggregateExecutionSimulation. Both run the same heap loop, ResourceDispatchEvent
and ResourceActivityEndEvent. Existing event type integers and serving trace
fields remain unchanged. AGGREGATE_SUBMIT (-1) queues owner readiness before
same-time dispatch; it does not reserve capacity. Activity ends (0) release
resources before dispatch (8). NativeResourceContext routes tagged AggregateOwner
completion to an exactly-once all-work record; stage/batch tuple keys still route
to the original serving callback. No aggregate path enters BatchStageEnd/BatchEnd.

Records include completion identity, activity intervals, makespan, summed service,
reservation utilization and native JSON/Chrome traces with cost provenance.
Makespan includes waiting, and differs from summed service when activities
overlap. TTFT, TPOT and token throughput are undefined for aggregates. No serving
objects or metric store are created by the aggregate entry. A time-limited partial
execution cannot produce a completion report; diagnostics retain partial evidence.
Empty work requires an explicit omission boundary. Physical placement, aggregate
memory, dynamic online admission and performance calibration are not provided.
