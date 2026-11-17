import pytest

from flymaze.topology import MazeEdge, MazeNode, MazeTopology, NodeKind


def test_topology_is_connected_and_digest_is_stable() -> None:
    topology = MazeTopology(
        nodes=(
            MazeNode("start", 0.0, 0.0, NodeKind.START),
            MazeNode("corner", 10.0, 0.0),
            MazeNode("food", 10.0, 10.0, NodeKind.FOOD),
        ),
        edges=(MazeEdge("start", "corner"), MazeEdge("corner", "food")),
    )
    assert topology.start.node_id == "start"
    assert topology.food.node_id == "food"
    assert topology.digest() == topology.digest()


def test_topology_rejects_diagonal_edges() -> None:
    with pytest.raises(ValueError, match="axis-aligned"):
        MazeTopology(
            nodes=(
                MazeNode("start", 0.0, 0.0, NodeKind.START),
                MazeNode("food", 10.0, 10.0, NodeKind.FOOD),
            ),
            edges=(MazeEdge("start", "food"),),
        )


def test_topology_rejects_unreachable_food() -> None:
    with pytest.raises(ValueError, match="reachable"):
        MazeTopology(
            nodes=(
                MazeNode("start", 0.0, 0.0, NodeKind.START),
                MazeNode("middle", 10.0, 0.0),
                MazeNode("food", 20.0, 0.0, NodeKind.FOOD),
            ),
            edges=(MazeEdge("start", "middle"),),
        )
