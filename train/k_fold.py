import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of 'results_correct_run2' to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_processing import load_data, get_labels
from model import BoneTumorClassifierResNet34
from dataset import CustomDataset
from imblearn.over_sampling import RandomOverSampler
from preprocessing import *

# Hyperparameters
batch_size = 128
num_classes = 3
num_epochs = 50
learning_rate = 0.001
# TO DO:
# Define data directory (where the images are stored)
data_dir = ''
# Define directory where the test set images are stored
test_data_dir = ''

# TO DO:
# Load data
# Define the path to the csv file containing the metadata for the cross validation
train_val_csv = ''
metadata = load_data(train_val_csv)
# Define the path to the csv fail containing the metadata for the test set
test_csv = ''
test_set = load_data(test_csv)

# Define the label map
label_map = {'benign': 0, 'intermediate': 1, 'malignant': 2}

# Optimal preprocessing and augmentation pipeline for the validation dataset
train_transform = transforms.Compose([
    remove_white_background,
    remove_black_background,
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224), 
    Sharpen(sharpness_factor=2.0),
    transforms.ToTensor(),
    normalize_image
])

# Optimal preprocessing and augmentation pipeline for the test set
test_transform = transforms.Compose([
    remove_white_background,
    remove_black_background,
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    Sharpen(sharpness_factor=2.0),
    transforms.ToTensor(),
    normalize_image
])

# In case you would want to try other preprocessing or augmentation methods, comment out those who you don't want to be included
dim = 224 # Define dimension for resizing
your_transform = transforms.Compose([
    remove_white_background,
    remove_black_background,
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    ResizeAndCrop(dim),
    ResizeAndPad(dim),
    baseline_resize,
    # Consider adapting the parameters as it may be dataset-specific
    BilateralFilter(diameter=5, sigma_color=100, sigma_space=100),
    histogram_equalization_with_clahe,
    HomomorphicFilter(cutoff=0.2 , alpha=5.0 , beta=0.5),
    Sharpen(sharpness_factor=2.0),
    apply_gaussian_blur,
    GammaCorrection(gamma=0.5),
    transforms.ToTensor(),
    normalize_image
])

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metrics storage
accuracies, f1_scores, roc_auc_scores = [], [], []
classwise_roc_auc_scores = {class_idx: [] for class_idx in label_map.values()}
best_model_weights, best_val_f1 = None, 0.0

# Split the indices of the DataFrame for k-fold cross-validation
for fold_idx, (train_index, val_index) in enumerate(kf.split(metadata), 1):
    print(f"Fold {fold_idx}/5")
    
    # Extract training and validation data for this fold
    train_data_fold = metadata.iloc[train_index]
    val_data_fold = metadata.iloc[val_index]

    train_labels_fold = get_labels(train_data_fold)
    val_labels_fold = get_labels(val_data_fold)

    # Perform oversampling on the training data for this fold
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(train_data_fold, train_labels_fold)
    train_data_fold = pd.DataFrame(X_resampled, columns=train_data_fold.columns)
    train_labels_fold = y_resampled

    # Create custom datasets for training and validation
    train_dataset_fold = CustomDataset(train_data_fold, train_labels_fold, data_dir=data_dir, transform=train_transform)
    val_dataset_fold = CustomDataset(val_data_fold, val_labels_fold, data_dir=data_dir, transform=test_transform)

    # Create data loaders for training and validation
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2)

    # Initialize model, loss function, and optimizer
    model = BoneTumorClassifierResNet34(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    val_epoch_accuracies = []
    val_epoch_f1_scores = []
    val_epoch_roc_auc_scores = []

    # Train the model using the current fold's data
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader_fold, desc=f'Epoch {epoch + 1}/{num_epochs} (Training)', disable=True):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate training loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_labels, all_val_preds, all_val_probs = [], [], []
        
        with torch.no_grad():
            for val_images, val_labels in tqdm(val_loader_fold, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)', disable=True):
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_images.size(0)
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

                all_val_labels.extend(val_labels.cpu().numpy())
                all_val_preds.extend(val_predicted.cpu().numpy())
                all_val_probs.extend(nn.functional.softmax(val_outputs, dim=1).cpu().numpy())

        # Calculate validation loss and metrics
        val_epoch_loss = val_running_loss / val_total
        val_losses.append(val_epoch_loss)

        val_epoch_acc = val_correct / val_total
        val_epoch_accuracies.append(val_epoch_acc)

        val_epoch_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_epoch_f1_scores.append(val_epoch_f1)

        # Calculate ROC-AUC scores
        val_epoch_roc_auc = roc_auc_score(np.eye(num_classes)[all_val_labels], np.array(all_val_probs), average=None)
        val_epoch_roc_auc_scores.append(val_epoch_roc_auc)

        print(f"Current epoch acc: {val_epoch_acc}, F1-score: {val_epoch_f1}, ROC-AUC: {val_epoch_roc_auc}")

    # Calculate metrics for this fold
    fold_accuracy = np.mean(val_epoch_accuracies)
    accuracies.append(fold_accuracy)
    fold_f1 = np.mean(val_epoch_f1_scores)
    f1_scores.append(fold_f1)
    fold_roc_auc = np.mean(val_epoch_roc_auc_scores)
    roc_auc_scores.append(fold_roc_auc)

     # Save the model if it has the best validation accuracy so far
    if fold_f1 > best_val_f1:
        print('Best model for fold: ', fold_idx)
        best_val_f1 = fold_f1
        best_model_weights = model.state_dict()   

    all_val_labels_np, all_val_probs_np = np.array(all_val_labels), np.array(all_val_probs)
    fpr, tpr, val_roc_auc = {}, {}, {}

    for label, idx in label_map.items():
        fpr[idx], tpr[idx], _ = roc_curve(all_val_labels_np == idx, all_val_probs_np[:, idx])
        val_roc_auc[idx] = auc(fpr[idx], tpr[idx])

    # Append ROC AUC scores to the respective lists in the dictionary
    for class_idx, score in val_roc_auc.items():
        classwise_roc_auc_scores[class_idx].append(score)

    # Plot and save the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(current_dir, f'loss_curves_fold_{fold_idx}.png'))
    plt.show()

# Compute mean and standard deviation of metrics across all folds
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_f1_score = np.mean(f1_scores)
std_f1_score = np.std(f1_scores)
mean_roc_auc = np.mean(roc_auc_scores, axis=0)
std_roc_auc = np.std(roc_auc_scores, axis=0)

# Compute mean and std for each class name
for class_name, class_idx in label_map.items():
    scores = classwise_roc_auc_scores[class_idx]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Class {class_name}: Mean ROC-AUC = {mean_score:.4f}, Std ROC-AUC = {std_score:.4f}")

# Print mean and standard deviation of accuracy, F1-score, and ROC-AUC score
print(f'Mean Accuracy across folds: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
print(f'Mean F1-score across folds: {mean_f1_score:.4f} ± {std_f1_score:.4f}')
print(f'Mean ROC-AUC across folds: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}')

## Testing

# Load the best model weights for final evaluation
model.load_state_dict(best_model_weights)

# Calculate confusion matrix for test set
from calculate_confusion_matrix import calculate_confusion_matrix
conf_matrix_test, test_true_labels, test_probs, test_predicted_labels = calculate_confusion_matrix(test_set, model, batch_size, device, test_data_dir, test_transform)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Set tick labels using the label map
tick_marks = np.arange(len(label_map))
plt.xticks(tick_marks, label_map.keys())
plt.yticks(tick_marks, label_map.keys())

# Add labels to each cell
thresh = conf_matrix_test.max() / 2.
for i, j in np.ndindex(conf_matrix_test.shape):
    plt.text(j, i, format(conf_matrix_test[i, j], '0.2f'),  # Use '0.2f' for floating-point numbers
         ha="center", va="center",
         color="white" if conf_matrix_test[i, j] > (2.0 * thresh) else "black")

plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.tight_layout()
plt.show()

# Save the plot as an image
plt.savefig(os.path.join(current_dir,'confusion_matrix_test.png'))

# Compute ROC AUC score for each class in the test set
roc_auc_scores = {}
for label, idx in label_map.items():
    fpr, tpr, _ = roc_curve(test_true_labels == idx, test_probs[:, idx])
    roc_auc_scores[label] = auc(fpr, tpr)

# Print ROC AUC scores
for label, score in roc_auc_scores.items():
    print(f'Test ROC AUC score for class {label}: {score:.4f}')

# Overall ROC AUC score
mean_roc_auc = np.mean(list(roc_auc_scores.values()))
print(f'Average Test ROC AUC score: {mean_roc_auc:.4f}')

# Compute ROC curve and ROC area for each class in the test set
fpr, tpr, roc_auc = {}, {}, {}

# Plot ROC curve
plt.figure(figsize=(10, 8))

for label, idx in label_map.items():
    fpr[idx], tpr[idx], _ = roc_curve(test_true_labels == idx, test_probs[:, idx])
    roc_auc[idx] = auc(fpr[idx], tpr[idx])

for label, idx in label_map.items():
    plt.plot(fpr[idx], tpr[idx], label=f'ROC curve (area = {roc_auc[idx]:0.2f}) for {label}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Test')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(current_dir,'roc_curve_test.png'))
plt.show()

# Calculate accuracy and F1 score on test set
accuracy_test = accuracy_score(test_true_labels, test_predicted_labels)
f1_test = f1_score(test_true_labels, test_predicted_labels, average='macro')

print(f'Average Test Accuracy: {accuracy_test:.4f}')
print(f'Average Test F1 Score: {f1_test:.4f}')