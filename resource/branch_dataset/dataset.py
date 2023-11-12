import os
import pandas as pd
import glob
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
 
# creating object for our image dataset
class BranchData(Dataset):
    # defining constructor
    def __init__(self, directory, transform=None):
        # directory containing the images
        self.directory = directory
        input_files = glob.glob(os.path.join(directory, "JPEGImages", "*.jpg"))
        self.image_names = [os.path.basename(path)[:-4] for path in input_files]

        # transform to be applied on images
        self.transform = transform
 
        # Number of images in dataset
        self.len = len(input_files)
 
    # getting the length
    def __len__(self):
        return len(self.image_names)
 
    # getting the data items
    def __getitem__(self, idx):
        # defining the image path
        image_name = self.image_names[idx]
        # reading the images
        input = read_image(os.path.join(self.directory, "JPEGImages", f"{image_name}.jpg"))
        output = read_image(os.path.join(self.directory, "SegmentationClass", f"{image_name}.png"))
 
        # apply the transform if not set to None
        if self.transform:
            input = self.transform(input)
            output *= 255
            output = self.transform(output)
        
        # returning the image and label
        return input, output