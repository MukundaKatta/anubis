"""Tests for anubis.core — graph, validation, and compilation."""

import pytest

from anubis.core import (
    AppCompiler,
    AppGraph,
    Component,
    Connection,
    GraphValidator,
    Port,
    PortDirection,
    PortType,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_in(name="input", required=True):
    return Port(name, PortType.TEXT, PortDirection.INPUT, required=required)

def _text_out(name="output"):
    return Port(name, PortType.TEXT, PortDirection.OUTPUT)

def _any_in(name="input", required=True):
    return Port(name, PortType.ANY, PortDirection.INPUT, required=required)

def _any_out(name="output"):
    return Port(name, PortType.ANY, PortDirection.OUTPUT)


def _simple_component(cid, in_port=None, out_port=None, ctype="generic"):
    ports = []
    if in_port:
        ports.append(in_port)
    if out_port:
        ports.append(out_port)
    return Component(id=cid, type=ctype, ports=ports)


# ---------------------------------------------------------------------------
# Port tests
# ---------------------------------------------------------------------------

class TestPort:
    def test_accepts_same_type(self):
        inp = _text_in()
        out = _text_out()
        assert inp.accepts(out) is True

    def test_rejects_different_type(self):
        inp = Port("in", PortType.EMBEDDING, PortDirection.INPUT)
        out = _text_out()
        assert inp.accepts(out) is False

    def test_any_type_accepts_all(self):
        inp = _any_in()
        out = _text_out()
        assert inp.accepts(out) is True

    def test_output_cannot_accept(self):
        out1 = _text_out("a")
        out2 = _text_out("b")
        assert out1.accepts(out2) is False


# ---------------------------------------------------------------------------
# Component tests
# ---------------------------------------------------------------------------

class TestComponent:
    def test_generate_id(self):
        id1 = Component.generate_id()
        id2 = Component.generate_id()
        assert id1 != id2
        assert len(id1) == 8

    def test_get_port(self):
        comp = _simple_component("c1", _text_in("data"), _text_out("result"))
        assert comp.get_port("data") is not None
        assert comp.get_port("missing") is None

    def test_input_output_ports(self):
        comp = _simple_component("c1", _text_in(), _text_out())
        assert len(comp.get_input_ports()) == 1
        assert len(comp.get_output_ports()) == 1


# ---------------------------------------------------------------------------
# AppGraph tests
# ---------------------------------------------------------------------------

class TestAppGraph:
    def test_add_and_get_component(self):
        g = AppGraph("test")
        comp = _simple_component("a", _text_in(), _text_out())
        g.add_component(comp)
        assert g.get_component("a") is comp

    def test_add_duplicate_raises(self):
        g = AppGraph()
        g.add_component(_simple_component("a"))
        with pytest.raises(ValueError):
            g.add_component(_simple_component("a"))

    def test_remove_component(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out()))
        g.add_component(_simple_component("b", in_port=_text_in()))
        g.connect("a", "output", "b", "input")
        g.remove_component("a")
        assert "a" not in g.components
        assert len(g.connections) == 0

    def test_connect_and_disconnect(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out()))
        g.add_component(_simple_component("b", in_port=_text_in()))
        g.connect("a", "output", "b", "input")
        assert len(g.connections) == 1
        assert g.disconnect("a", "output", "b", "input") is True
        assert len(g.connections) == 0

    def test_connect_missing_component_raises(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out()))
        with pytest.raises(KeyError):
            g.connect("a", "output", "missing", "input")

    def test_connect_missing_port_raises(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out()))
        g.add_component(_simple_component("b", in_port=_text_in()))
        with pytest.raises(ValueError):
            g.connect("a", "bad_port", "b", "input")

    def test_upstream_downstream(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out()))
        g.add_component(_simple_component("b", in_port=_text_in(), out_port=_text_out("out")))
        g.connect("a", "output", "b", "input")
        assert g.get_upstream("b") == ["a"]
        assert g.get_downstream("a") == ["b"]

    def test_entry_and_terminal_points(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out()))
        g.add_component(_simple_component("b", in_port=_text_in(), out_port=_text_out("out")))
        g.add_component(_simple_component("c", in_port=_text_in()))
        g.connect("a", "output", "b", "input")
        g.connect("b", "out", "c", "input")
        assert g.entry_points() == ["a"]
        assert g.terminal_points() == ["c"]


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestGraphValidator:
    def _build_linear_graph(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out()))
        g.add_component(_simple_component("b", in_port=_text_in(), out_port=_text_out("out")))
        g.connect("a", "output", "b", "input")
        return g

    def test_valid_graph_no_errors(self):
        g = self._build_linear_graph()
        errors = GraphValidator().validate(g)
        hard = [e for e in errors if e.severity == "error"]
        assert len(hard) == 0

    def test_cycle_detected(self):
        g = AppGraph()
        g.add_component(_simple_component("a", _text_in(), _text_out()))
        g.add_component(_simple_component("b", _text_in(), _text_out()))
        g.connect("a", "output", "b", "input")
        g.connect("b", "output", "a", "input")
        errors = GraphValidator().validate(g)
        cycle_errors = [e for e in errors if "Cycle" in e.message]
        assert len(cycle_errors) >= 1

    def test_type_mismatch(self):
        g = AppGraph()
        a = Component("a", "x", ports=[Port("out", PortType.EMBEDDING, PortDirection.OUTPUT)])
        b = Component("b", "y", ports=[Port("in", PortType.TEXT, PortDirection.INPUT)])
        g.add_component(a)
        g.add_component(b)
        g.connect("a", "out", "b", "in")
        errors = GraphValidator().validate(g)
        mismatch = [e for e in errors if "mismatch" in e.message.lower()]
        assert len(mismatch) == 1

    def test_duplicate_connection_warning(self):
        g = self._build_linear_graph()
        g.connect("a", "output", "b", "input")  # duplicate
        errors = GraphValidator().validate(g)
        dups = [e for e in errors if "Duplicate" in e.message]
        assert len(dups) == 1


# ---------------------------------------------------------------------------
# Compiler tests
# ---------------------------------------------------------------------------

class TestAppCompiler:
    def test_compile_linear(self):
        g = AppGraph()
        g.add_component(_simple_component("a", out_port=_text_out(), ctype="prompt"))
        g.add_component(_simple_component("b", in_port=_text_in(), ctype="output"))
        g.connect("a", "output", "b", "input")
        steps = AppCompiler().compile(g)
        assert len(steps) == 2
        assert steps[0].component_id == "a"
        assert steps[1].component_id == "b"

    def test_compile_rejects_cycle(self):
        g = AppGraph()
        g.add_component(_simple_component("a", _text_in(), _text_out()))
        g.add_component(_simple_component("b", _text_in(), _text_out()))
        g.connect("a", "output", "b", "input")
        g.connect("b", "output", "a", "input")
        with pytest.raises(ValueError, match="validation failed"):
            AppCompiler().compile(g)

    def test_compiled_step_input_mappings(self):
        g = AppGraph()
        g.add_component(_simple_component("src", out_port=_text_out(), ctype="prompt"))
        g.add_component(_simple_component("dst", in_port=_text_in(), ctype="output"))
        g.connect("src", "output", "dst", "input")
        steps = AppCompiler().compile(g)
        dst_step = steps[1]
        assert "input" in dst_step.input_mappings
        assert dst_step.input_mappings["input"] == ("src", "output")
