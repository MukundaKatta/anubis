# 🔱 Anubis — Visual LLM App Builder

> **Egyptian Mythology**: Guide to the Afterlife | Design LLM applications as component graphs

[![GitHub Pages](https://img.shields.io/badge/🌐_Live_Demo-Visit_Site-blue?style=for-the-badge)](https://MukundaKatta.github.io/anubis/)
[![GitHub](https://img.shields.io/github/license/MukundaKatta/anubis?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/MukundaKatta/anubis?style=flat-square)](https://github.com/MukundaKatta/anubis/stargazers)

## Overview

Anubis is a visual LLM application builder that lets you design RAG pipelines,
multi-agent conversations, and chatbots as directed component graphs. Wire
together Prompt, LLM, RAG, Router, Memory, and Output components — Anubis
validates the graph and compiles it into an executable pipeline.

**Tech Stack:** Python 3.9+ | No external dependencies

## Quick Start

```bash
git clone https://github.com/MukundaKatta/anubis.git
cd anubis
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Architecture

```
src/anubis/
├── core.py          # AppGraph, Port, Component, GraphValidator, AppCompiler
├── components.py    # PromptComponent, LLMComponent, RAGComponent, RouterComponent, etc.
└── runtime.py       # AppRuntime, RequestContext, RuntimeMetrics
```

**Component Graph** — each node is a `Component` with typed input/output `Port`s.
Connections form a DAG that the `GraphValidator` checks for cycles, type
mismatches, and unconnected required ports. The `AppCompiler` topologically
sorts the graph into a `CompiledStep` list, and the `AppRuntime` executes it
while collecting latency and token-usage metrics.

### Built-in Components

| Component | Inputs | Outputs | Purpose |
|-----------|--------|---------|---------|
| **Prompt** | text, variables | text | Template rendering |
| **LLM** | prompt | response, token_usage | Language model completion |
| **RAG** | query | context, documents | Retrieval-augmented generation |
| **Router** | text | route, output | Keyword-based routing |
| **Memory** | text | history, output | Conversation history |
| **Output** | any | result | Terminal collector |

## Usage Example

```python
from anubis.components import (
    create_prompt_component,
    create_llm_component,
    create_output_component,
)
from anubis.core import AppGraph
from anubis.runtime import AppRuntime

graph = AppGraph("My Chatbot")
prompt = create_prompt_component("Answer: {input}", component_id="prompt")
llm = create_llm_component("gpt-4", component_id="llm")
out = create_output_component("text", component_id="out")

graph.add_component(prompt)
graph.add_component(llm)
graph.add_component(out)
graph.connect("prompt", "output", "llm", "prompt")
graph.connect("llm", "response", "out", "input")

runtime = AppRuntime()
ctx = runtime.execute(graph, {"input": "What is RAG?"})
print(ctx.component_outputs["out"]["result"])
```

## Testing

```bash
PYTHONPATH=src python3 -m pytest tests/ -v --tb=short
```

50 tests covering core graph operations, all six component processors,
factory functions, validation rules, compilation, and end-to-end runtime
execution.

## Live Demo

Visit the landing page: **https://MukundaKatta.github.io/anubis/**

## License

MIT License — see [LICENSE](LICENSE) for details.

## Part of the Mythological Portfolio

Project **#anubis** in the [Mythological Portfolio](https://github.com/MukundaKatta) by Officethree Technologies.
