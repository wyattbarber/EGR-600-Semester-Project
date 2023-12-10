import numpy as np
from typing import List, Tuple
from .node import Node


def identify_branches(skeleton: np.ndarray) -> List[Node]:
    """
    Identifies nodes in a mask.

    Identifies locations where multiple lines meet, or one line ends.

    :param ndarray skeleton: Skeletonized binary mask

    :return: All identified node descriptors
    :rtype: List[Node]
    """
    out = []
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y,x]:
                n = count_adjacent_pixels(skeleton, (y, x))
                if n != 2:
                    out.append(
                        Node(x, y)
                    )
    return out


def count_adjacent_pixels(mask: np.ndarray, pos: Tuple[int, int]):
    min_x = max(0, pos[1]-1)
    max_x = min(mask.shape[1]-1, pos[1]+1)
    min_y = max(0, pos[0]-1)
    max_y = min(mask.shape[0]-1, pos[0]+1)

    sum = 0
    for x in range(min_x, max_x+1):
        for y in range(min_y, max_y+1):
            if ((y, x) != pos) and mask[y,x]:
                sum += 1
    return sum


def set_node_width(img: np.ndarray, node: Node):
    r = [0, 0, 0, 0]
    while(img[node.pixel_coord[0] + r[0], node.pixel_coord[1]] != 0):
        r[0] += 1
        if (node.pixel_coord[0] + r[0]) >= img.shape[0]:
            break
    while(img[node.pixel_coord[0] - r[1], node.pixel_coord[1]] != 0):
        r[1] += 1
        if (node.pixel_coord[0] - r[1]) < 0:
            break
    while(img[node.pixel_coord[0], node.pixel_coord[1] + r[2]] != 0):
        r[2] += 1
        if (node.pixel_coord[1] + r[2]) >= img.shape[1]:
            break
    while(img[node.pixel_coord[0], node.pixel_coord[1] - r[3]] != 0):
        r[3] += 1
        if (node.pixel_coord[1] - r[3]) < img.shape[1]:
            break
    node._set_width(int(np.median(r)))


def get_circular_indices(shape: Tuple[int, int], pos: Tuple[int, int], r: int):
    step = np.pi / (2 * r)
    angles = np.linspace(-np.pi, np.pi, int(2 * np.pi / step))
    pts = [(int(pos[0] + r*np.sin(a)), int(pos[1] + r*np.cos(a))) for a in angles]
    
    out = np.zeros(shape, dtype=bool)
    for pt in pts:
        if (pt[0] >= 0) and (pt[0] < shape[0]) and (pt[1] >= 1) and (pt[1] < shape[1]) and (pt not in out):
            out[pt[0],pt[1]] = True

