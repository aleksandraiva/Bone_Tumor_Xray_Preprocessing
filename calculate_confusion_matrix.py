import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from dataset import CustomDataset 
from data_processing import get_labels

def calculate_confusion_matrix(data, model, batch_size, device, data_dir, transform):
    # Initialize test dataset and data loader
    labels = get_labels(data)
    dataset = CustomDataset(data, labels, data_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()

    true_labels = []
    predicted_labels = []
    test_probs = []

    with torch.no_grad():
        for test_images, labels in tqdm(loader, desc='Testing'):
            test_images = test_images.to(device)
            labels = labels.to(device)

            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs, 1)

            # Append true and predicted labels for this batch
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(test_predicted.cpu().numpy())

            # Calculate softmax probabilities
            softmax_probs = torch.softmax(test_outputs, dim=1)

            # Append probabilities for this batch
            test_probs.extend(softmax_probs.cpu().numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Convert list of true labels and probabilities to numpy arrays
    true_labels = np.array(true_labels)
    test_probs = np.array(test_probs)
    predicted_labels = np.array(predicted_labels)

    return conf_matrix, true_labels, test_probs, predicted_labels
