import torch
import os
from PIL import Image

# Define a custom dataset class passing the metadata, the encoded labels, 
# the data directory and transformations (preprocessing, augmenation...)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, data_dir, transform=None):
        self.data = data
        self.labels = labels
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data.iloc[idx]['image'])

        img = Image.open(img_path)


        # Convert image to RGB if it has 4 channels
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        label = self.labels[idx]  # Use numerical label

        if self.transform:
            img = self.transform(img)

        return img, label
    
    # Get the original image from the dataset
    
    def get_original(self, idx):
        img_path = os.path.join(self.data_dir, self.data.iloc[idx]['image'])
        img = Image.open(img_path)

        # Convert image to RGB if it has 4 channels
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        return img
    
    # Find minimal dimension of an image in the dataset

    def find_min_dimension(self):
        min_dimension = float('inf')
        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.data_dir, row['image'])
            with Image.open(img_path) as img:
                width, height = img.size
                min_dimension = min(min_dimension, min(width, height))
        return min_dimension

