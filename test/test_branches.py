import pytest
import numpy as np
from src.branches import identify_branches, set_node_width
from src.node import Node


@pytest.mark.parametrize("img, coord",[
    (np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]), [(2, 3), (6, 1), (0, 3), (7, 2)]),
])
def test_branch_points(img, coord):
    nodes = identify_branches(img)
    pts = [n.pixel_coord for n in nodes]
    assert len(pts) == 6, "Unexpected number of nodes found"
    for pt in coord:
        assert pt in pts, "Expected point not found"


@pytest.mark.parametrize("skeleton,img,coord,width",[
    (np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]), np.array([
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
    ]), (2, 3), 2),
    (np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]), np.array([
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
    ]), (7, 2), 2),
])
def test_width(skeleton, img, coord, width):
    nodes = identify_branches(skeleton)
    n = [n for n in nodes if (n.pixel_coord == coord)][0]
    set_node_width(img, n)
    assert n.width == width, "Width of node calculated incorrectly"