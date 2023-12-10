from .node import Node
import numpy as np
from typing import List, Tuple
import cv2
from .branches import count_adjacent_pixels

def set_neighbors_corr(skeleton: np.ndarray, node: Node, others: List[Node], tol: int = 3, delete: bool = False):
    scores = []
    for n in others:
        if n is not node:
            w = abs(node.pixel_coord[1] - n.pixel_coord[1])
            h = abs(node.pixel_coord[0] - n.pixel_coord[0])
            kernel = cv2.UMat(np.zeros((h, w)))

            if (node.pixel_coord[1] >= n.pixel_coord[1]) and (node.pixel_coord[0] >= n.pixel_coord[0]):
                pt0 = (0, 0)
                pt1 = (h-1, w-1)
            elif (node.pixel_coord[1] >= n.pixel_coord[1]) and (node.pixel_coord[0] < n.pixel_coord[0]):
                pt0 = (h-1, 0)
                pt1 = (0, w-1)
            elif (node.pixel_coord[1] < n.pixel_coord[1]) and (node.pixel_coord[0] < n.pixel_coord[0]):
                pt0 = (h-1, w-1)
                pt1 = (0, 0)
            else:
                pt0 = (0, w-1)
                pt1 = (h-1, 0)

            cx = node.pixel_coord[1] + n.pixel_coord[1] // 2
            cy = node.pixel_coord[0] + n.pixel_coord[0] // 2
            xmin = cx - (w//2)
            xmax = cx + (w//2) + 1
            ymin = cy - (h//2) - 1
            ymax = cy + (h//2) 
            sub_img = np.zeros((h, w))
            for x in range(xmin, xmax+1):
                for y in range(ymin, ymax+1):
                    try:
                        sub_img[y - ymin, x - xmin] = skeleton[y,x]
                    except IndexError:
                        pass
            
            cv2.line(kernel, pt0, pt1, color=1, thickness=tol)
            score = np.sum(kernel.get() * sub_img)
            scores.append(score)
        else:
            scores.append(0)

    N = len(node.angles)
    idx = np.argsort(scores)[-N:]
    node._add_children(*[others[i] for i in idx])
        
    if delete:
        for i in idx:
            del others[i]


def set_neighbor_angle(node: Node, others: List[Node], i: int, w: float = 0.5):
    scores = []
    for n in others:
        if n is not node:
            w = abs(node.pixel_coord[1] - n.pixel_coord[1])
            h = abs(node.pixel_coord[0] - n.pixel_coord[0])
            d = np.sqrt(w**2 + h**2)

            x = node.pixel_coord[1] - (d * np.sin(node.angles[i]))
            y = node.pixel_coord[1] - (d * np.sin(node.angles[i]))

            score = 1 / np.sqrt((x - n.pixel_coord[1])**2 + (y - n.pixel_coord[0])**2)
            if np.isnan(score):
                score = 1e6
            score *= w
            score += (1 - w) * (1 / d)
            scores.append(score)
        else:
            scores.append(0)
        
    idx = np.argsort(scores)[-1]
    node._add_children(others[idx])


def set_neighbors_trace(skeleton: np.ndarray, node: Node, others: List[Node]):
    node._add_children(*[_trace_line(skeleton, point, node.pixel_coord, others) for point in _get_adjacent_points(skeleton, node.pixel_coord)])

def _get_adjacent_points(skeleton, point):
    min_x = max(0, point[1]-1)
    max_x = min(skeleton.shape[1]-1, point[1]+1)
    min_y = max(0, point[0]-1)
    max_y = min(skeleton.shape[0]-1, point[0]+1)

    out = []
    for x in range(min_x, max_x+1):
        for y in range(min_y, max_y+1):
            if ((y, x) != point) and skeleton[y,x]:
                out.append((y, x))
    return out


def _trace_line(skeleton: np.ndarray, point: Tuple[int, int], prev: Tuple[int, int], nodes: List[Node]):
    if count_adjacent_pixels(skeleton, point) != 2:
        try:
            return [n for n in nodes if n.pixel_coord == point][0]
        except IndexError:
            raise RuntimeError(f"Trace ended at {point} but no node found at that location")
    else:
        points = [p for p in _get_adjacent_points(skeleton, point) if p != prev]
        return _trace_line(skeleton, points[0], point, nodes)



