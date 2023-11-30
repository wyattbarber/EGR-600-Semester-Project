import numpy as np
from typing import Tuple, List


class Node:
    """
    Class for representing one node of a modeled tree.

    """
    _x: int # Horizontal pixel coordinates
    _y: int # Vertical pixel coordinates
    _width: int # Width of the node
    _angles: List[float] # Angles of attached branches
    _children: List # List of attached nodes

    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y
        self._children = []
    
    @property
    def pixel_coord(self) -> Tuple[int, int]:
        return (self._y, self._x)
    
    @property
    def width(self):
        return self._width

    def _set_width(self, width: int):
        self._width = width

    @property
    def angles(self):
        return self._angles
    
    def _set_angles(self, angles):
        self._angles = angles

    @property
    def children(self):
        return self._children
    
    def _add_children(self, *nodes):
        self._children.extend(nodes)