from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import numpy as np

def compute_confusion_matrix(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32).unsqueeze(1)  # Ensure the correct shape
            #masks = masks.to(device)
            outputs = model(images)

            preds = torch.sigmoid(outputs) > 0.5  # Apply threshold to get binary predictions

            preds = preds.int()

            masks = masks.int()

            all_predictions.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(masks.view(-1).cpu().numpy())

    # Generate confusion matrix

    cm = confusion_matrix(all_targets, all_predictions)
    return cm

"""
To plot CM
conf_matrix = compute_confusion_matrix(model, test_loader, device)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32).unsqueeze(1)  # Ensure the correct shape
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Apply threshold to get binary predictions
            # Convert predictions to integer format
            preds = preds.int()
            # Ensure targets are also in integer format
            masks = masks.int()
            all_predictions.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(masks.view(-1).cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    return accuracy, precision, recall, f1