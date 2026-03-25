"""Core graph system for Anubis visual LLM application builder.

Provides the foundational data structures and logic for building,
validating, and compiling component graphs into executable pipelines.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class PortDirection(Enum):
    """Direction of a component port."""
    INPUT = "input"
    OUTPUT = "output"


class PortType(Enum):
    """Data type that flows through a port."""
    TEXT = "text"
    EMBEDDING = "embedding"
    DOCUMENT = "document"
    MESSAGE = "message"
    ANY = "any"


@dataclass
class Port:
    """An input or output port on a component."""
    name: str
    port_type: PortType
    direction: PortDirection
    required: bool = True

    def accepts(self, other: Port) -> bool:
        """Check if this input port accepts data from the given output port."""
        if self.direction != PortDirection.INPUT:
            return False
        if other.direction != PortDirection.OUTPUT:
            return False
        if self.port_type == PortType.ANY or other.port_type == PortType.ANY:
            return True
        return self.port_type == other.port_type


@dataclass
class Component:
    """A single node in the application graph."""
    id: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    ports: List[Port] = field(default_factory=list)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique component ID."""
        return str(uuid.uuid4())[:8]

    def get_input_ports(self) -> List[Port]:
        """Return all input ports."""
        return [p for p in self.ports if p.direction == PortDirection.INPUT]

    def get_output_ports(self) -> List[Port]:
        """Return all output ports."""
        return [p for p in self.ports if p.direction == PortDirection.OUTPUT]

    def get_port(self, name: str) -> Optional[Port]:
        """Look up a port by name."""
        for p in self.ports:
            if p.name == name:
                return p
        return None


@dataclass
class Connection:
    """A directed edge between two component ports."""
    source_id: str
    source_port: str
    target_id: str
    target_port: str

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (self.source_id, self.source_port, self.target_id, self.target_port)


class AppGraph:
    """Directed graph of components with typed, named ports."""

    def __init__(self, name: str = "Untitled App") -> None:
        self.name = name
        self.components: Dict[str, Component] = {}
        self.connections: List[Connection] = []

    # -- mutators ----------------------------------------------------------

    def add_component(self, component: Component) -> None:
        """Add a component to the graph."""
        if component.id in self.components:
            raise ValueError(f"Component '{component.id}' already exists")
        self.components[component.id] = component

    def remove_component(self, component_id: str) -> None:
        """Remove a component and all its connections."""
        if component_id not in self.components:
            raise KeyError(f"Component '{component_id}' not found")
        del self.components[component_id]
        self.connections = [
            c for c in self.connections
            if c.source_id != component_id and c.target_id != component_id
        ]

    def connect(
        self,
        source_id: str,
        source_port: str,
        target_id: str,
        target_port: str,
    ) -> Connection:
        """Create a connection between two component ports."""
        if source_id not in self.components:
            raise KeyError(f"Source component '{source_id}' not found")
        if target_id not in self.components:
            raise KeyError(f"Target component '{target_id}' not found")

        src_comp = self.components[source_id]
        tgt_comp = self.components[target_id]

        src_port = src_comp.get_port(source_port)
        if src_port is None:
            raise ValueError(
                f"Port '{source_port}' not found on component '{source_id}'"
            )
        tgt_port = tgt_comp.get_port(target_port)
        if tgt_port is None:
            raise ValueError(
                f"Port '{target_port}' not found on component '{target_id}'"
            )

        conn = Connection(source_id, source_port, target_id, target_port)
        self.connections.append(conn)
        return conn

    def disconnect(self, source_id: str, source_port: str, target_id: str, target_port: str) -> bool:
        """Remove a specific connection. Returns True if found and removed."""
        key = (source_id, source_port, target_id, target_port)
        for i, c in enumerate(self.connections):
            if c.key == key:
                self.connections.pop(i)
                return True
        return False

    # -- queries -----------------------------------------------------------

    def get_component(self, component_id: str) -> Component:
        if component_id not in self.components:
            raise KeyError(f"Component '{component_id}' not found")
        return self.components[component_id]

    def get_upstream(self, component_id: str) -> List[str]:
        """Return IDs of components that feed into the given component."""
        return list({
            c.source_id for c in self.connections if c.target_id == component_id
        })

    def get_downstream(self, component_id: str) -> List[str]:
        """Return IDs of components that the given component feeds into."""
        return list({
            c.target_id for c in self.connections if c.source_id == component_id
        })

    def entry_points(self) -> List[str]:
        """Components with no incoming connections (graph roots)."""
        targets = {c.target_id for c in self.connections}
        return [cid for cid in self.components if cid not in targets]

    def terminal_points(self) -> List[str]:
        """Components with no outgoing connections (graph leaves)."""
        sources = {c.source_id for c in self.connections}
        return [cid for cid in self.components if cid not in sources]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """A single validation issue."""
    severity: str  # "error" or "warning"
    message: str
    component_id: Optional[str] = None


class GraphValidator:
    """Validates an AppGraph for structural and type correctness."""

    def validate(self, graph: AppGraph) -> List[ValidationError]:
        errors: List[ValidationError] = []
        errors.extend(self._check_cycles(graph))
        errors.extend(self._check_unconnected_required_ports(graph))
        errors.extend(self._check_type_mismatches(graph))
        errors.extend(self._check_duplicate_connections(graph))
        return errors

    # -- individual checks -------------------------------------------------

    def _check_cycles(self, graph: AppGraph) -> List[ValidationError]:
        """Detect cycles via DFS."""
        errors: List[ValidationError] = []
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def _dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for downstream in graph.get_downstream(node):
                if downstream in in_stack:
                    errors.append(ValidationError(
                        "error",
                        f"Cycle detected involving '{node}' -> '{downstream}'",
                        component_id=node,
                    ))
                    return True
                if downstream not in visited:
                    if _dfs(downstream):
                        return True
            in_stack.discard(node)
            return False

        for cid in graph.components:
            if cid not in visited:
                _dfs(cid)
        return errors

    def _check_unconnected_required_ports(self, graph: AppGraph) -> List[ValidationError]:
        errors: List[ValidationError] = []
        connected_inputs: Set[Tuple[str, str]] = set()
        connected_outputs: Set[Tuple[str, str]] = set()
        for conn in graph.connections:
            connected_inputs.add((conn.target_id, conn.target_port))
            connected_outputs.add((conn.source_id, conn.source_port))

        for cid, comp in graph.components.items():
            for port in comp.get_input_ports():
                if port.required and (cid, port.name) not in connected_inputs:
                    # entry-point components are allowed to have unconnected inputs
                    if cid not in graph.entry_points():
                        errors.append(ValidationError(
                            "warning",
                            f"Required input port '{port.name}' on '{cid}' is unconnected",
                            component_id=cid,
                        ))
        return errors

    def _check_type_mismatches(self, graph: AppGraph) -> List[ValidationError]:
        errors: List[ValidationError] = []
        for conn in graph.connections:
            src = graph.components.get(conn.source_id)
            tgt = graph.components.get(conn.target_id)
            if src is None or tgt is None:
                continue
            src_port = src.get_port(conn.source_port)
            tgt_port = tgt.get_port(conn.target_port)
            if src_port is None or tgt_port is None:
                continue
            if not tgt_port.accepts(src_port):
                errors.append(ValidationError(
                    "error",
                    (
                        f"Type mismatch: '{conn.source_id}.{conn.source_port}' "
                        f"({src_port.port_type.value}) -> "
                        f"'{conn.target_id}.{conn.target_port}' "
                        f"({tgt_port.port_type.value})"
                    ),
                    component_id=conn.target_id,
                ))
        return errors

    def _check_duplicate_connections(self, graph: AppGraph) -> List[ValidationError]:
        errors: List[ValidationError] = []
        seen: Set[Tuple[str, str, str, str]] = set()
        for conn in graph.connections:
            if conn.key in seen:
                errors.append(ValidationError(
                    "warning",
                    f"Duplicate connection {conn.key}",
                ))
            seen.add(conn.key)
        return errors


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@dataclass
class CompiledStep:
    """One step in the execution plan."""
    component_id: str
    component_type: str
    config: Dict[str, Any]
    input_mappings: Dict[str, Tuple[str, str]]  # port_name -> (source_id, source_port)


class AppCompiler:
    """Converts a validated AppGraph into an ordered execution plan."""

    def __init__(self, validator: Optional[GraphValidator] = None) -> None:
        self.validator = validator or GraphValidator()

    def compile(self, graph: AppGraph) -> List[CompiledStep]:
        """Topologically sort the graph and produce an execution plan.

        Raises ValueError if the graph contains errors.
        """
        errors = self.validator.validate(graph)
        hard_errors = [e for e in errors if e.severity == "error"]
        if hard_errors:
            messages = "; ".join(e.message for e in hard_errors)
            raise ValueError(f"Graph validation failed: {messages}")

        order = self._topological_sort(graph)
        steps: List[CompiledStep] = []
        for cid in order:
            comp = graph.components[cid]
            input_map: Dict[str, Tuple[str, str]] = {}
            for conn in graph.connections:
                if conn.target_id == cid:
                    input_map[conn.target_port] = (conn.source_id, conn.source_port)
            steps.append(CompiledStep(
                component_id=cid,
                component_type=comp.type,
                config=dict(comp.config),
                input_mappings=input_map,
            ))
        return steps

    @staticmethod
    def _topological_sort(graph: AppGraph) -> List[str]:
        """Kahn's algorithm."""
        in_degree: Dict[str, int] = {cid: 0 for cid in graph.components}
        for conn in graph.connections:
            if conn.target_id in in_degree:
                in_degree[conn.target_id] += 1

        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            queue.sort()  # deterministic ordering
            node = queue.pop(0)
            order.append(node)
            for conn in graph.connections:
                if conn.source_id == node and conn.target_id in in_degree:
                    in_degree[conn.target_id] -= 1
                    if in_degree[conn.target_id] == 0:
                        queue.append(conn.target_id)
        return order
