import numpy as np
from .branches import identify_branches, set_node_width
from .node import Node
from typing import List


def assemble_nodes(skeleton: np.ndarray, img: np.ndarray) -> List[Node]:
    nodes = identify_branches(skeleton)
    for n in nodes:
        set_node_width(img, n)
    return nodes