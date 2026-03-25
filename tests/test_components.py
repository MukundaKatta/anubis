"""Tests for anubis.components — built-in component processors and factories."""

import pytest

from anubis.components import (
    LLMProcessor,
    MemoryProcessor,
    OutputProcessor,
    PromptProcessor,
    RAGProcessor,
    RouterProcessor,
    create_llm_component,
    create_memory_component,
    create_output_component,
    create_processor,
    create_prompt_component,
    create_rag_component,
    create_router_component,
)
from anubis.core import PortDirection


class TestPromptProcessor:
    def test_simple_template(self):
        p = PromptProcessor()
        result = p.process({"input": "hello"}, {"template": "Say: {input}"})
        assert result["output"] == "Say: hello"

    def test_default_template(self):
        p = PromptProcessor()
        result = p.process({"input": "world"}, {})
        assert result["output"] == "world"

    def test_template_with_variables(self):
        p = PromptProcessor()
        result = p.process(
            {"input": "question", "variables": {"tone": "polite"}},
            {"template": "Please answer this {input} in a {tone} way"},
        )
        assert "polite" in result["output"]


class TestLLMProcessor:
    def test_returns_response(self):
        p = LLMProcessor()
        result = p.process({"prompt": "hi"}, {"model": "gpt-test", "temperature": 0.5, "max_tokens": 100})
        assert "gpt-test" in result["response"]
        assert "hi" in result["response"]
        assert isinstance(result["token_usage"], int)

    def test_default_config(self):
        p = LLMProcessor()
        result = p.process({"prompt": "hello world"}, {})
        assert "default-model" in result["response"]


class TestRAGProcessor:
    def test_returns_documents(self):
        p = RAGProcessor()
        result = p.process({"query": "test"}, {"top_k": 2, "collection": "wiki"})
        assert len(result["documents"]) == 2
        assert "wiki" in result["context"]

    def test_default_top_k(self):
        p = RAGProcessor()
        result = p.process({"query": "q"}, {})
        assert len(result["documents"]) == 3


class TestRouterProcessor:
    def test_routes_by_keyword(self):
        rules = [{"keyword": "help", "route": "support"}, {"keyword": "buy", "route": "sales"}]
        p = RouterProcessor()
        result = p.process({"input": "I need help"}, {"rules": rules, "default_route": "general"})
        assert result["route"] == "support"

    def test_default_route(self):
        p = RouterProcessor()
        result = p.process({"input": "random text"}, {"rules": [], "default_route": "fallback"})
        assert result["route"] == "fallback"


class TestOutputProcessor:
    def test_text_format(self):
        p = OutputProcessor()
        result = p.process({"input": "data"}, {"format": "text"})
        assert result["result"] == "data"

    def test_json_format(self):
        p = OutputProcessor()
        result = p.process({"input": "data"}, {"format": "json"})
        assert result["result"]["format"] == "json"


class TestMemoryProcessor:
    def test_stores_messages(self):
        p = MemoryProcessor()
        p.process({"input": "hello"}, {"max_turns": 5, "role": "user"})
        result = p.process({"input": "world"}, {"max_turns": 5, "role": "user"})
        assert len(result["history"]) == 2

    def test_max_turns_truncation(self):
        p = MemoryProcessor()
        for i in range(5):
            p.process({"input": f"msg{i}"}, {"max_turns": 3, "role": "user"})
        result = p.process({"input": "final"}, {"max_turns": 3, "role": "user"})
        assert len(result["history"]) == 3

    def test_clear(self):
        p = MemoryProcessor()
        p.process({"input": "a"}, {"role": "user"})
        p.clear()
        result = p.process({"input": "b"}, {"role": "user"})
        assert len(result["history"]) == 1


# ---------------------------------------------------------------------------
# Factory / registry tests
# ---------------------------------------------------------------------------

class TestComponentFactories:
    def test_create_prompt_component(self):
        c = create_prompt_component("Hello {input}")
        assert c.type == "prompt"
        assert c.config["template"] == "Hello {input}"
        assert any(p.direction == PortDirection.INPUT for p in c.ports)

    def test_create_llm_component(self):
        c = create_llm_component("my-model")
        assert c.config["model"] == "my-model"

    def test_create_rag_component(self):
        c = create_rag_component("docs", top_k=5)
        assert c.config["top_k"] == 5

    def test_create_router_component(self):
        c = create_router_component([{"keyword": "hi", "route": "greet"}])
        assert len(c.config["rules"]) == 1

    def test_create_output_component(self):
        c = create_output_component("json")
        assert c.config["format"] == "json"

    def test_create_memory_component(self):
        c = create_memory_component(max_turns=20)
        assert c.config["max_turns"] == 20

    def test_create_processor_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_processor("nonexistent")
