import cv2
import numpy as np
from .node import Node
from .branches import count_adjacent_pixels
from typing import List

def set_node_angles(skeleton: np.ndarray, node: Node, kernel_size: int = 9):
    angles = get_angles(skeleton, node, kernel_size)
    node._set_angles(angles)

def get_angles(skeleton: np.ndarray, node: Node, kernel_size: int = 9) -> List[float]:
    """
    Get the angles that branches are attached to a node at.

    Angles are measured in radians, with 0 being straight up.

    :param np.ndarray skeleton: Skeletonized binary mask.
    :param kernel_size: Size of the mask used to detect line orientation. Must be odd, default 5.
    :type kernel_size: int, optional

    :param Node node: Node descriptor indicating a point in skeleton where there is a branch, bend, or endpoint.
    """
    N = count_adjacent_pixels(skeleton, node.pixel_coord)
    K = kernel_size**2

    cx = node.pixel_coord[1]
    cy = node.pixel_coord[0]
    xmin = cx - (kernel_size//2)
    xmax = cx + (kernel_size//2) + 1
    ymin = cy - (kernel_size//2) - 1
    ymax = cy + (kernel_size//2) 
    sub_img = np.zeros((kernel_size, kernel_size))
    for x in range(xmin, xmax+1):
        for y in range(ymin, ymax+1):
            try:
                sub_img[y - ymin, x - xmin] = skeleton[y,x]
            except IndexError:
                pass

    corr = _correlate_angles(sub_img, K, kernel_size)
    idx_max = np.argsort(corr)
    return [i * ((2*np.pi)/K) for i in idx_max[-N:]]


def _correlate_angles(skeleton: np.ndarray, K: int, kernel_size: int):
    corr = []
    for mask in _make_rotating_masks(K, kernel_size):
        corr.append(
            np.sum(skeleton * mask)
        )
    return corr


def _make_rotating_masks(N: int, kernel_size: int):
    mask = np.zeros((kernel_size, kernel_size))
    mask = np.zeros((kernel_size, kernel_size))
    mask[
        :(kernel_size//2)+1,
        (kernel_size//2)-1:(kernel_size//2)+2
    ] = 1

    out = []
    for k in range(N):
        _tf = cv2.getRotationMatrix2D(
            (kernel_size/2, kernel_size/2),
            k * (360/N),
            1.0
        )
        _mask = cv2.warpAffine(mask, _tf, mask.shape)
        out.append(_mask)

    return out
