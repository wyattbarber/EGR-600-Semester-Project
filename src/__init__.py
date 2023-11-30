import numpy as np
from .branches import identify_branches, set_node_width
from .orientation import set_node_angles
from .connectivity import set_neighbor_angle, set_neighbors_corr
from .node import Node
from typing import List


def assemble_nodes(skeleton: np.ndarray, img: np.ndarray) -> List[Node]:
    nodes = identify_branches(skeleton)
    for n in nodes:
        set_node_width(img, n)
        set_node_angles(skeleton, n)
        # set_neighbors_corr(skeleton, n, nodes)
        for i in range(len(n.angles)):
            set_neighbor_angle(n, nodes, i)
    return nodes