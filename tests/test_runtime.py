"""Tests for anubis.runtime — execution engine and metrics."""

import pytest

from anubis.components import (
    create_llm_component,
    create_output_component,
    create_prompt_component,
    create_rag_component,
)
from anubis.core import AppGraph
from anubis.runtime import AppRuntime, RequestContext, RuntimeMetrics


class TestRuntimeMetrics:
    def test_record_accumulates(self):
        m = RuntimeMetrics()
        m.record("a", 0.1, 10)
        m.record("b", 0.2, 20)
        assert m.total_tokens == 30
        assert abs(m.total_latency - 0.3) < 1e-9
        assert m.component_latencies["a"] == 0.1


class TestRequestContext:
    def test_get_output(self):
        ctx = RequestContext(request_id="r1", initial_input={})
        ctx.component_outputs["comp1"] = {"text": "hello"}
        assert ctx.get_output("comp1", "text") == "hello"
        assert ctx.get_output("comp1", "missing") is None
        assert ctx.get_output("no_comp", "text") is None


class TestAppRuntime:
    def _build_prompt_llm_output_graph(self):
        g = AppGraph("test-app")
        prompt = create_prompt_component("Translate: {input}", component_id="prompt")
        llm = create_llm_component("test-model", component_id="llm")
        out = create_output_component("text", component_id="out")
        g.add_component(prompt)
        g.add_component(llm)
        g.add_component(out)
        g.connect("prompt", "output", "llm", "prompt")
        g.connect("llm", "response", "out", "input")
        return g

    def test_execute_pipeline(self):
        g = self._build_prompt_llm_output_graph()
        rt = AppRuntime()
        ctx = rt.execute(g, {"input": "hello"}, request_id="test-1")
        assert ctx.request_id == "test-1"
        # Output component should have produced a result
        assert "out" in ctx.component_outputs
        assert ctx.component_outputs["out"]["result"] is not None

    def test_metrics_collected(self):
        g = self._build_prompt_llm_output_graph()
        rt = AppRuntime()
        ctx = rt.execute(g, {"input": "hello"})
        metrics = ctx.metadata.get("metrics")
        assert metrics is not None
        assert "prompt" in metrics.component_latencies
        assert "llm" in metrics.component_latencies
        assert metrics.total_latency > 0

    def test_rag_pipeline(self):
        g = AppGraph("rag-app")
        rag = create_rag_component("knowledge", top_k=2, component_id="rag")
        llm = create_llm_component("rag-model", component_id="llm")
        out = create_output_component("text", component_id="out")
        g.add_component(rag)
        g.add_component(llm)
        g.add_component(out)
        g.connect("rag", "context", "llm", "prompt")
        g.connect("llm", "response", "out", "input")
        rt = AppRuntime()
        ctx = rt.execute(g, {"query": "what is AI?"})
        assert "out" in ctx.component_outputs

    def test_single_component(self):
        g = AppGraph()
        prompt = create_prompt_component("{input}!", component_id="p")
        g.add_component(prompt)
        rt = AppRuntime()
        ctx = rt.execute(g, {"input": "hi"})
        assert ctx.component_outputs["p"]["output"] == "hi!"

    def test_empty_graph(self):
        g = AppGraph()
        rt = AppRuntime()
        ctx = rt.execute(g, {})
        assert ctx.component_outputs == {}
