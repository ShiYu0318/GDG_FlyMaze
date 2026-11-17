"""Maze topology independent of MuJoCo geometry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum


class NodeKind(StrEnum):
    START = "start"
    WAYPOINT = "waypoint"
    JUNCTION = "junction"
    DEAD_END = "dead_end"
    FOOD = "food"


@dataclass(frozen=True, slots=True)
class MazeNode:
    node_id: str
    x_mm: float
    y_mm: float
    kind: NodeKind = NodeKind.WAYPOINT


@dataclass(frozen=True, slots=True)
class MazeEdge:
    node_a: str
    node_b: str

    def __post_init__(self) -> None:
        if self.node_a == self.node_b:
            raise ValueError("maze edge endpoints must differ")


@dataclass(frozen=True, slots=True)
class MazeTopology:
    """Validated axis-aligned graph with one start and one food node."""

    nodes: tuple[MazeNode, ...]
    edges: tuple[MazeEdge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("maze topology requires nodes")
        by_id = {node.node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("maze node IDs must be unique")
        coordinates = {(node.x_mm, node.y_mm) for node in self.nodes}
        if len(coordinates) != len(self.nodes):
            raise ValueError("maze node coordinates must be unique")
        starts = [node for node in self.nodes if node.kind is NodeKind.START]
        foods = [node for node in self.nodes if node.kind is NodeKind.FOOD]
        if len(starts) != 1 or len(foods) != 1:
            raise ValueError("maze topology requires exactly one start and one food node")
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in by_id}
        seen_edges: set[frozenset[str]] = set()
        for edge in self.edges:
            if edge.node_a not in by_id or edge.node_b not in by_id:
                raise ValueError("maze edge references an unknown node")
            key = frozenset((edge.node_a, edge.node_b))
            if key in seen_edges:
                raise ValueError("maze edges must be unique")
            seen_edges.add(key)
            first = by_id[edge.node_a]
            second = by_id[edge.node_b]
            if first.x_mm != second.x_mm and first.y_mm != second.y_mm:
                raise ValueError("MVP maze edges must be axis-aligned")
            adjacency[edge.node_a].add(edge.node_b)
            adjacency[edge.node_b].add(edge.node_a)
        reachable = {starts[0].node_id}
        frontier = [starts[0].node_id]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] - reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
        if foods[0].node_id not in reachable:
            raise ValueError("food node must be reachable from start")

    @property
    def start(self) -> MazeNode:
        return next(node for node in self.nodes if node.kind is NodeKind.START)

    @property
    def food(self) -> MazeNode:
        return next(node for node in self.nodes if node.kind is NodeKind.FOOD)

    def digest(self) -> str:
        payload = {
            "nodes": [
                {"id": node.node_id, "x": node.x_mm, "y": node.y_mm, "kind": node.kind}
                for node in sorted(self.nodes, key=lambda value: value.node_id)
            ],
            "edges": sorted([sorted((edge.node_a, edge.node_b)) for edge in self.edges]),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
