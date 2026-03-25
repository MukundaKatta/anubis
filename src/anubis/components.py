"""Built-in components for the Anubis visual LLM application builder.

Each component defines its input/output ports and a process() method
that transforms data flowing through the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from anubis.core import Component, Port, PortDirection, PortType


# ---------------------------------------------------------------------------
# Base helper
# ---------------------------------------------------------------------------

def _make_component(
    component_type: str,
    ports: List[Port],
    config: Optional[Dict[str, Any]] = None,
    component_id: Optional[str] = None,
) -> Component:
    """Convenience factory for creating a Component with auto-generated id."""
    return Component(
        id=component_id or Component.generate_id(),
        type=component_type,
        config=config or {},
        ports=list(ports),
    )


# ---------------------------------------------------------------------------
# Component processors
# ---------------------------------------------------------------------------

class ComponentProcessor:
    """Base class for component processing logic."""

    def process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PromptProcessor(ComponentProcessor):
    """Formats a prompt template with input variables."""

    def process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        template = config.get("template", "{input}")
        text = inputs.get("input", "")
        variables = inputs.get("variables", {})
        if not isinstance(variables, dict):
            variables = {}
        merged = {"input": text, **variables}
        try:
            rendered = template.format(**merged)
        except KeyError:
            rendered = template.replace("{input}", str(text))
        return {"output": rendered}


class LLMProcessor(ComponentProcessor):
    """Simulates LLM completion (returns a deterministic echo for testing)."""

    def process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get("prompt", "")
        model = config.get("model", "default-model")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 256)
        # Simulated response for testing / offline use
        response_text = f"[{model}@t={temperature}] Response to: {prompt}"
        token_count = max(1, len(prompt.split()))
        return {
            "response": response_text,
            "token_usage": min(token_count, max_tokens),
        }


class RAGProcessor(ComponentProcessor):
    """Simulates retrieval-augmented generation."""

    def process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query", "")
        top_k = config.get("top_k", 3)
        collection = config.get("collection", "default")
        # Simulate retrieved documents
        docs = [
            {"content": f"Document {i+1} for '{query}' from {collection}", "score": round(1.0 - i * 0.1, 2)}
            for i in range(top_k)
        ]
        context = "\n".join(d["content"] for d in docs)
        return {"context": context, "documents": docs}


class RouterProcessor(ComponentProcessor):
    """Routes input to one of several named outputs based on rules."""

    def process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        text = str(inputs.get("input", ""))
        rules = config.get("rules", [])
        default_route = config.get("default_route", "default")
        selected = default_route
        for rule in rules:
            keyword = rule.get("keyword", "")
            route = rule.get("route", default_route)
            if keyword and keyword.lower() in text.lower():
                selected = route
                break
        return {"route": selected, "output": text}


class OutputProcessor(ComponentProcessor):
    """Terminal component that collects final output."""

    def process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        fmt = config.get("format", "text")
        data = inputs.get("input", "")
        if fmt == "json":
            return {"result": {"data": data, "format": "json"}}
        return {"result": str(data)}


class MemoryProcessor(ComponentProcessor):
    """Maintains conversation history across invocations."""

    def __init__(self) -> None:
        self._history: List[Dict[str, str]] = []

    def process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        message = inputs.get("input", "")
        max_turns = config.get("max_turns", 10)
        role = config.get("role", "user")
        if message:
            self._history.append({"role": role, "content": str(message)})
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]
        return {
            "history": list(self._history),
            "output": message,
        }

    def clear(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Registry and factories
# ---------------------------------------------------------------------------

PROCESSOR_REGISTRY: Dict[str, type] = {
    "prompt": PromptProcessor,
    "llm": LLMProcessor,
    "rag": RAGProcessor,
    "router": RouterProcessor,
    "output": OutputProcessor,
    "memory": MemoryProcessor,
}


def create_processor(component_type: str) -> ComponentProcessor:
    """Instantiate a processor for the given component type."""
    cls = PROCESSOR_REGISTRY.get(component_type)
    if cls is None:
        raise ValueError(f"Unknown component type: {component_type}")
    return cls()


# ---------------------------------------------------------------------------
# Pre-built component factories
# ---------------------------------------------------------------------------

def create_prompt_component(template: str = "{input}", component_id: Optional[str] = None) -> Component:
    return _make_component("prompt", [
        Port("input", PortType.TEXT, PortDirection.INPUT),
        Port("variables", PortType.ANY, PortDirection.INPUT, required=False),
        Port("output", PortType.TEXT, PortDirection.OUTPUT),
    ], {"template": template}, component_id)


def create_llm_component(model: str = "default-model", component_id: Optional[str] = None) -> Component:
    return _make_component("llm", [
        Port("prompt", PortType.TEXT, PortDirection.INPUT),
        Port("response", PortType.TEXT, PortDirection.OUTPUT),
        Port("token_usage", PortType.ANY, PortDirection.OUTPUT),
    ], {"model": model, "temperature": 0.7, "max_tokens": 256}, component_id)


def create_rag_component(collection: str = "default", top_k: int = 3, component_id: Optional[str] = None) -> Component:
    return _make_component("rag", [
        Port("query", PortType.TEXT, PortDirection.INPUT),
        Port("context", PortType.TEXT, PortDirection.OUTPUT),
        Port("documents", PortType.ANY, PortDirection.OUTPUT),
    ], {"collection": collection, "top_k": top_k}, component_id)


def create_router_component(rules: Optional[List[Dict[str, str]]] = None, component_id: Optional[str] = None) -> Component:
    return _make_component("router", [
        Port("input", PortType.TEXT, PortDirection.INPUT),
        Port("output", PortType.TEXT, PortDirection.OUTPUT),
        Port("route", PortType.ANY, PortDirection.OUTPUT),
    ], {"rules": rules or [], "default_route": "default"}, component_id)


def create_output_component(fmt: str = "text", component_id: Optional[str] = None) -> Component:
    return _make_component("output", [
        Port("input", PortType.ANY, PortDirection.INPUT),
        Port("result", PortType.ANY, PortDirection.OUTPUT),
    ], {"format": fmt}, component_id)


def create_memory_component(max_turns: int = 10, component_id: Optional[str] = None) -> Component:
    return _make_component("memory", [
        Port("input", PortType.TEXT, PortDirection.INPUT),
        Port("output", PortType.TEXT, PortDirection.OUTPUT),
        Port("history", PortType.ANY, PortDirection.OUTPUT),
    ], {"max_turns": max_turns, "role": "user"}, component_id)
