import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(csv_file):
    "Load metadata from csv file"
    data = pd.read_csv(csv_file,  encoding='Windows-1252', delimiter=',')
    return data

def split_data(data, test_size=0.1, val_size=0.1):
    """Split data into train, validation, and test sets"""
    # First, split data into train and test
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, stratify=data['malignancy'], shuffle=True)
    
    # Then, split train data into train and validation
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42, stratify=train_data['malignancy'], shuffle=True)

    return train_data, val_data, test_data

# Extract only the necceassary columns from the metadata
def extract_data(data):
    return data.loc[:, ['image', 'malignancy']]

# Get list of the encoded labels corresponding to the images from the metadata
def get_labels(data):
    # Create a label map
    label_map = {'benign': 0, 'intermediate': 1, 'malignant': 2}

    # Initialize lists to store converted labels
    labels = []

    # Convert string labels to numerical values using the label map
    for _, row in data.iterrows():
        labels.append(label_map[row['malignancy']])

    return labels



