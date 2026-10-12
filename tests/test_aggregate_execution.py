"""Independent synthetic timelines through the actual native event heap."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from vidur.aggregate_execution import AggregateExecutionSimulation, AggregateSubmission
from vidur.entities.execution_plan import (
    ExecutionActivity as A,
)
from vidur.entities.execution_plan import (
    ExecutionPlan as P,
)
from vidur.entities.execution_plan import (
    MemoryAccess,
)
from vidur.event_loop import EventLoop
from vidur.scheduler.execution_plan_builder import (
    ActivityDeclaration,
    BoundCost,
    ExecutionPlanBuilder,
    PlanDeclaration,
    SelectedRequest,
    SelectedWork,
)
from vidur.scheduler.native_resource_context import AggregateOwner
from vidur.simulator import Simulator
from vidur.types import EventType


def activity(name, seconds, resources=("R0",), dependencies=()):
    return A(
        name,
        seconds,
        resources,
        dependencies,
        cost_provenance="Synthetic seconds, not calibration",
        cost_granularity="One synthetic activity",
        plan_id="synthetic-plan",
    )


def fixture(capacity=1, owner=None):
    plan = P(
        (
            activity("A", 3),
            activity("B", 2, ("R1",)),
            activity("C", 1, dependencies=("A", "B")),
            activity("D", 2),
        )
    )
    return AggregateExecutionSimulation(
        {"R0": capacity, "R1": 1},
        (AggregateSubmission(owner or AggregateOwner("slice", "slice_done"), plan),),
    )


class AggregateExecutionTests(unittest.TestCase):
    def test_actual_heap_timelines_and_no_serving_objects_or_hooks(self):
        for capacity, expected, intervals in (
            (1, 6, [("A", 0, 3), ("B", 0, 2), ("C", 3, 4), ("D", 4, 6)]),
            (2, 4, [("A", 0, 3), ("B", 0, 2), ("C", 3, 4), ("D", 0, 2)]),
        ):
            with (
                self.subTest(capacity=capacity),
                patch(
                    "vidur.entities.Request.__init__",
                    side_effect=AssertionError("request"),
                ),
                patch(
                    "vidur.entities.Batch.__init__", side_effect=AssertionError("batch")
                ),
                patch(
                    "vidur.entities.Cluster.__init__",
                    side_effect=AssertionError("cluster"),
                ),
                patch(
                    "vidur.metrics.MetricsStore.__init__",
                    side_effect=AssertionError("metrics"),
                ),
                patch(
                    "vidur.events.batch_end_event.BatchEndEvent.handle_event",
                    side_effect=AssertionError("token completion"),
                ),
            ):
                sim = fixture(capacity)
                sim.run()
                report = sim.report()
            result = report["completions"][0]
            self.assertEqual(result["makespan_seconds"], expected)
            self.assertEqual(result["total_service_seconds"], 8)
            self.assertEqual(
                [
                    (a["activity_id"], a["start_seconds"], a["finish_seconds"])
                    for a in result["activities"]
                ],
                intervals,
            )
            self.assertEqual(report["resource_used"], {"R0": 0, "R1": 0})
            self.assertEqual(len(report["completions"]), 1)
            ends = [e for e in report["events"] if "activity" in e]
            self.assertEqual(len(ends), 4)
            self.assertTrue(all(e["work_id"] == "slice" for e in ends))
            self.assertTrue(
                all("batch_id" not in e and "stage_id" not in e for e in ends)
            )
            self.assertTrue({"TTFT", "TPOT"} <= set(report["unsupported_metrics"]))
            self.assertTrue(issubclass(Simulator, EventLoop))
            self.assertIs(AggregateExecutionSimulation.run, EventLoop.run)
            sim.run()
            self.assertEqual(sim.report(), report)
            other = fixture(capacity)
            other.run()
            self.assertEqual(other.report(), report)

    def test_separate_owners_contend_without_duplicate_completion(self):
        plan = P((activity("work", 3),))
        sim = AggregateExecutionSimulation(
            {"R0": 1},
            tuple(
                AggregateSubmission(AggregateOwner(str(i), f"done_{i}"), plan)
                for i in range(2)
            ),
        )
        sim.run()
        results = sim.report()["completions"]
        self.assertEqual([r["completed_at_seconds"] for r in results], [3, 6])
        self.assertEqual([r["activities"][0]["start_seconds"] for r in results], [0, 3])
        with self.assertRaisesRegex(RuntimeError, "twice"):
            sim.context.on_native_plan_complete(AggregateOwner("0", "done_0"), 6)

    def test_ready_times_do_not_reserve_future_capacity(self):
        sim = AggregateExecutionSimulation(
            {"R0": 1, "R1": 1},
            (
                AggregateSubmission(
                    AggregateOwner("first", "first_done"),
                    P(
                        (
                            activity("wait", 4, ("R1",)),
                            activity("join", 1, dependencies=("wait",)),
                        )
                    ),
                ),
                AggregateSubmission(
                    AggregateOwner("later", "later_done"), P((activity("later", 2),)), 1
                ),
            ),
        )
        sim.run()
        results = sim.report()["completions"]
        self.assertEqual([r["completed_at_seconds"] for r in results], [5, 3])
        self.assertEqual(results[1]["makespan_seconds"], 2)

    def test_zero_costs_end_before_dispatch_and_atomic_multi_resource_reservations(
        self,
    ):
        sim = AggregateExecutionSimulation(
            {"R0": 1, "R1": 1},
            (
                AggregateSubmission(
                    AggregateOwner("slice", "done"),
                    P(
                        (
                            activity("link", 2, ("R1",)),
                            activity("both", 3, ("R0", "R1")),
                            activity("zero", 0, dependencies=("both",)),
                        )
                    ),
                ),
            ),
        )
        sim.run()
        r = sim.report()
        self.assertEqual(r["completions"][0]["makespan_seconds"], 5)
        self.assertEqual(r["completions"][0]["activities"][1]["start_seconds"], 2)
        self.assertEqual(
            sum(
                e["event_type"] == int(EventType.RESOURCE_ACTIVITY_END)
                for e in r["events"]
            ),
            3,
        )
        at_two = [e["event_type"] for e in r["events"] if e["time"] == 2]
        self.assertEqual(at_two, [0, 8])

    def test_explicit_empty_boundary_and_zero_window(self):
        sim = AggregateExecutionSimulation(
            {"R0": 1},
            (
                AggregateSubmission(
                    AggregateOwner("empty", "done"), P((), ("Explicit omitted work",))
                ),
            ),
        )
        sim.run()
        result = sim.report()["completions"][0]
        self.assertEqual(result["makespan_seconds"], 0)
        self.assertIsNone(result["resource_reservation_utilization"]["R0"])

    def test_invalid_programs_and_memory_fail_before_queueing(self):
        good = P((activity("work", 1),))
        bad_plans = [
            P(()),
            P((activity("a", 1, dependencies=("later",)),)),
            P((activity("a", 1, ("unknown",)),)),
            P((activity("a", float("nan")),)),
            P((A("a", 1, ("R0",)),)),
        ]
        for plan in bad_plans:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                AggregateExecutionSimulation(
                    {"R0": 1}, (AggregateSubmission(AggregateOwner("a", "done"), plan),)
                )
        for submissions in (
            (),
            [],
            (AggregateSubmission(AggregateOwner("a", "done"), good),) * 2,
        ):
            with self.assertRaises(ValueError):
                AggregateExecutionSimulation({"R0": 1}, submissions)
        for owner in (AggregateOwner("a", "other"), AggregateOwner("b", "done")):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                AggregateExecutionSimulation(
                    {"R0": 1},
                    (
                        AggregateSubmission(AggregateOwner("a", "done"), good),
                        AggregateSubmission(owner, good),
                    ),
                )
        with self.assertRaisesRegex(NotImplementedError, "memory"):
            AggregateExecutionSimulation(
                {"R0": 1},
                (
                    AggregateSubmission(
                        AggregateOwner("a", "done"),
                        P(
                            (
                                replace(
                                    good.activities[0],
                                    memory_reads=(MemoryAccess(0, "kv", 16),),
                                ),
                            )
                        ),
                    ),
                ),
            )
        for capacity in (0, -1, True):
            with self.assertRaises(ValueError):
                AggregateExecutionSimulation(
                    {"R0": capacity},
                    (AggregateSubmission(AggregateOwner("a", "done"), good),),
                )

    def test_partial_execution_cannot_report_completion(self):
        sim = AggregateExecutionSimulation(
            {"R0": 1},
            (
                AggregateSubmission(
                    AggregateOwner("a", "done"), P((activity("work", 3),))
                ),
            ),
            time_limit=2,
        )
        sim.run()
        self.assertEqual(sim.diagnostics()["current_time_seconds"], 2)
        self.assertTrue(sim.diagnostics()["terminated_at_time_limit"])
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            sim.report()
        self.assertEqual(sim.diagnostics()["resource_used"], {"R0": 1})

    def test_request_free_materialization_and_serving_validation_are_separate(self):
        decl = PlanDeclaration(
            (ActivityDeclaration("a", "a", ("R0",)),), plan_id="identity"
        )
        costs = {"a": BoundCost(1, "Synthetic", "Activity")}
        plan = ExecutionPlanBuilder.build_declaration(decl, costs, {"R0": 1})
        self.assertEqual(plan.activities[0].plan_id, "identity")
        with self.assertRaisesRegex(ValueError, "missing cost"):
            ExecutionPlanBuilder.build_declaration(decl, {}, {"R0": 1})
        memory_decl = replace(
            decl,
            activities=(
                replace(decl.activities[0], memory_reads=(MemoryAccess(7, "kv", 16),)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "owner validator"):
            ExecutionPlanBuilder.build_declaration(memory_decl, costs, {"R0": 1})

        class Policy:
            def declare(self, work):
                return memory_decl

            def resource_capacities(self):
                return {"R0": 1}

            def memory_pool_capacities(self):
                return {"kv": 16}

        class Costs:
            def bind(self, work):
                return costs

        builder = ExecutionPlanBuilder(Policy(), Costs())
        work = SelectedWork((SelectedRequest(7, 9, 8, True, 1),), 0)
        self.assertEqual(
            builder.build(work).activities[0].memory_reads, (MemoryAccess(7, "kv", 16),)
        )
        with self.assertRaisesRegex(ValueError, "selected work"):
            builder.build(SelectedWork((SelectedRequest(8, 9, 8, True, 1),), 0))


if __name__ == "__main__":
    unittest.main()
