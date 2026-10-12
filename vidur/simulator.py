import atexit
import json

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.event_loop import EventLoop
from vidur.events import RequestArrivalEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry

logger = init_logger(__name__)


class Simulator(EventLoop):
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config

        super().__init__(
            config.time_limit,
            config.metrics_config.write_json_trace,
            config.metrics_config.enable_chrome_trace,
        )

        self._cluster = Cluster(
            self._config.cluster_config,
            self._config.metrics_config,
            self._config.request_generator_config,
        )
        self._metric_store = MetricsStore(self._config)
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )

        self._init_event_queue()
        atexit.register(self._write_output)

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._scheduler

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store

    def run(self) -> None:
        logger.info(
            f"Starting simulation with cluster: {self._cluster} and {len(self._event_queue)} requests"
        )

        # Retain the live serving trace flags, including bare-loop test callers.
        self._write_json_trace = self._config.metrics_config.write_json_trace
        self._enable_chrome_trace = self._config.metrics_config.enable_chrome_trace
        super().run()
        if not self._terminate and any(
            not context.aggregates_complete()
            for context in getattr(self, "_aggregate_contexts", ())
        ):
            raise RuntimeError("simulation has unfinished aggregate work with no events")

        logger.info(f"Simulation ended at: {self._time}s")

    def _write_output(self) -> None:
        logger.info("Writing output")

        self._metric_store.plot()
        logger.info("Metrics written")

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def submit_aggregate(self, replica_id, submission):
        """Opt-in resource-only work sharing a native replica's executor/heap."""
        from vidur.aggregate_execution import enqueue_aggregate

        replica = self._scheduler.get_replica_scheduler(replica_id)
        context = replica.native_resource_context
        if context is None:
            raise ValueError("aggregate submission requires native resource execution")
        enqueue_aggregate(self, context, submission, replica_id)
        if not hasattr(self, "_aggregate_contexts"):
            self._aggregate_contexts = set()
        self._aggregate_contexts.add(context)

    def _init_event_queue(self) -> None:
        requests = self._request_generator.generate()

        for request in requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._terminate = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)
