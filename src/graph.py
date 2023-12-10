from .node import Node
import numpy as np
from .branches import identify_branches, set_node_width
from .orientation import set_node_angles
from .connectivity import set_neighbor_angle, set_neighbors_corr, set_neighbors_trace
from .node import Node
from .cnn import ImageSegmentation
import torch
from torch import nn
import torchvision.transforms as T
from skimage.morphology import skeletonize
from typing import List
import cv2

class Tree:
    """
    Constructs a model of a tree from an image.
    """

    _nodes: List[Node]

    def __init__(self, image: torch.Tensor, model: ImageSegmentation):
        """
        Constructs a new model from an input image.

        :param image: RGB image of a tree.
        :param ImageSegmentation model: CNN to segment image to tree/not tree pixels.
        """
        self._size = (image.shape[-2], image.shape[-1])
        
        tf_in = T.Resize((128, 128))
        img_in = tf_in(image).float()
        if len(img_in.shape) < 4:
            img_in = img_in.unsqueeze(0)
        mask = model(img_in)

        tf_out = T.Resize(self._size)
        mask = tf_out(mask).detach().squeeze(0).permute(1, 2, 0).numpy()

        mask[mask > 0.5] = 1
        mask[mask < 1] = 0
        self._mask = mask
        
        skeleton = skeletonize(mask)
        self._skel = skeleton

        self._nodes = identify_branches(skeleton)
        for n in self._nodes:
            set_node_width(mask, n)
            set_node_angles(skeleton, n)
            #set_neighbors_corr(skeleton, n, self._nodes, tol=1)
            # for i in range(len(n.angles)):
            #     set_neighbor_angle(n, self._nodes, i, w=0.01)
            set_neighbors_trace(skeleton, n, self._nodes)

    def reconstruct_mask(self):
        """
        Reconstruct an approximation of the true mask based on the created model.

        :return: Binary 128x128 image
        :rtype: numpy.ndarray
        """
        out = np.zeros((*self._size, 3))
        for branch in self.edges:
            w = (branch[0].width + branch[1].width) // 2
            cv2.line(out, (branch[0].pixel_coord[1], branch[0].pixel_coord[0]), (branch[1].pixel_coord[1], branch[1].pixel_coord[0]), color=(1,1,1), thickness=w)
        return out[:,:,0]
    
    @property
    def edges(self):
        """
        List all branches in the model.

        :return: Pairs of (parent node, child node) branch connections.
        :rtype: List[Tuple[Node, Node]]
        """
        out = []
        for n in self._nodes:
            for c in n.children:
                out.append((n, c))
        return out
    
    @property
    def root(self):
        root_location = (self._size[0], self._size[1]/2)
        root_dist = 0
        _root = None
        for node in self._nodes:
            dist = np.sqrt(
                (node.pixel_coord[0] - root_location[0])**2 +
                (node.pixel_coord[1] - root_location[1])**2
            )
            if (_root is None) or (dist < root_dist):
                _root = node
                root_dist = dist            
        return _root

    @property
    def nodes(self):
        return self._nodes
    
    @property
    def confidence(self):
        return np.mean([node.confidence() for node in self._nodes])
