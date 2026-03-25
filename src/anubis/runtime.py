"""Runtime engine for executing compiled Anubis application graphs.

Handles request context propagation, component execution, and metrics
collection across the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from anubis.components import ComponentProcessor, create_processor
from anubis.core import AppCompiler, AppGraph, CompiledStep


@dataclass
class RuntimeMetrics:
    """Tracks latency and token usage per component."""
    component_latencies: Dict[str, float] = field(default_factory=dict)
    component_token_usage: Dict[str, int] = field(default_factory=dict)
    total_latency: float = 0.0
    total_tokens: int = 0

    def record(self, component_id: str, latency: float, tokens: int = 0) -> None:
        self.component_latencies[component_id] = latency
        self.component_token_usage[component_id] = tokens
        self.total_latency += latency
        self.total_tokens += tokens


@dataclass
class RequestContext:
    """Carries data through the execution pipeline."""
    request_id: str
    initial_input: Dict[str, Any]
    component_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_output(self, component_id: str, port_name: str) -> Any:
        """Retrieve a specific port value produced by a component."""
        outputs = self.component_outputs.get(component_id, {})
        return outputs.get(port_name)


class AppRuntime:
    """Executes a compiled app graph against a request context."""

    def __init__(self) -> None:
        self._processors: Dict[str, ComponentProcessor] = {}

    def execute(
        self,
        graph: AppGraph,
        initial_input: Dict[str, Any],
        request_id: str = "req-001",
        compiler: Optional[AppCompiler] = None,
    ) -> RequestContext:
        """Run the full pipeline and return the populated context."""
        compiler = compiler or AppCompiler()
        steps = compiler.compile(graph)
        metrics = RuntimeMetrics()

        ctx = RequestContext(
            request_id=request_id,
            initial_input=initial_input,
        )

        for step in steps:
            processor = self._get_processor(step)
            inputs = self._resolve_inputs(step, ctx)
            t0 = time.monotonic()
            outputs = processor.process(inputs, step.config)
            elapsed = time.monotonic() - t0

            tokens = outputs.get("token_usage", 0)
            if not isinstance(tokens, int):
                tokens = 0
            metrics.record(step.component_id, elapsed, tokens)
            ctx.component_outputs[step.component_id] = outputs

        ctx.metadata["metrics"] = metrics
        return ctx

    # -- internals ---------------------------------------------------------

    def _get_processor(self, step: CompiledStep) -> ComponentProcessor:
        """Get or create a processor for the step."""
        if step.component_id not in self._processors:
            self._processors[step.component_id] = create_processor(step.component_type)
        return self._processors[step.component_id]

    @staticmethod
    def _resolve_inputs(step: CompiledStep, ctx: RequestContext) -> Dict[str, Any]:
        """Build the input dict for a step from upstream outputs or initial input."""
        inputs: Dict[str, Any] = {}
        for port_name, (src_id, src_port) in step.input_mappings.items():
            value = ctx.get_output(src_id, src_port)
            if value is not None:
                inputs[port_name] = value
        # If this is an entry point with no resolved inputs, use initial_input
        if not inputs and not step.input_mappings:
            inputs = dict(ctx.initial_input)
        return inputs
