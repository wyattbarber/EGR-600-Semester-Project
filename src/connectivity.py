from .node import Node
import numpy as np
from typing import List
import cv2

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
    node._add_children(others[i] for i in idx)
        
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
        
    N = len(node.angles)
    idx = np.argsort(scores)[-N:]
    node._add_children(others[i] for i in idx)
