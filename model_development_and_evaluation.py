####################################
# Model development and evaluation #
####################################

from __future__ import annotations

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
import seaborn as sns

################################
# Data load and preparation    #
################################

DATA_ROOT = Path("/Users/saujanyathapaliya/Documents/breastcancer/processed_png/dataset_splits")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

CHECKPOINT_DIR = Path("/Users/saujanyathapaliya/Documents/breastcancer/processed_png/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = CHECKPOINT_DIR / "training_metrics.csv"
if METRICS_PATH.exists():
    METRICS_PATH.unlink()

for path in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if not path.exists():
        raise FileNotFoundError(f"Required directory missing: {path}")

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transforms)
val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=eval_transforms)
test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_transforms)

print(f"Classes: {train_dataset.classes}")

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

###################
# Build the model #
###################

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=1),
            nn.LeakyReLU(0.1),
        )

        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.feature_extractor(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

model = CustomModel()

#################
# Run the model #
#################

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
best_val_accuracy = 0.0
best_checkpoint_path = CHECKPOINT_DIR / "model_best.pth"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted_labels = (outputs > 0.5).float()
        predicted_labels_int = predicted_labels.view(-1).long()
        total_correct += (predicted_labels_int == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = total_correct / total_samples
    train_loss = running_loss / len(train_loader)

    model.eval()
    val_running_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs.squeeze(), val_labels.float())
            val_running_loss += val_loss.item()

            val_predicted_labels = (val_outputs > 0.5).float()
            val_predicted_labels_int = val_predicted_labels.view(-1).long()
            val_total_correct += (val_predicted_labels_int == val_labels).sum().item()
            val_total_samples += val_labels.size(0)

    val_accuracy = val_total_correct / val_total_samples
    val_loss = val_running_loss / len(val_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    )
    with open(METRICS_PATH, "a") as f:
        if f.tell() == 0:
            f.write("epoch,train_accuracy,val_accuracy,train_loss,val_loss\n")
        f.write(f"{epoch + 1},{train_accuracy:.6f},{val_accuracy:.6f},{train_loss:.6f},{val_loss:.6f}\n")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_checkpoint_path)
        print(f"New best checkpoint saved: {best_checkpoint_path} (val_acc={val_accuracy:.4f})")

####################
# Model evaluation #
####################

if not best_checkpoint_path.exists():
    raise FileNotFoundError(f"Best checkpoint not found at {best_checkpoint_path}. Train the model first.")

model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
model.eval()

true_labels = []
predicted_probs = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = model(test_images)
        predicted_probs.extend(outputs.cpu().numpy())
        true_labels.extend(test_labels.cpu().numpy())

true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)
predicted_labels = (predicted_probs > 0.5).astype(int)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

######################
# Confidence Intervals
######################

n_iterations = 1000
accuracy_bootstrap = []
precision_bootstrap = []
recall_bootstrap = []
f1_bootstrap = []
roc_auc_bootstrap = []

for _ in range(n_iterations):
    bootstrap_indices = np.random.choice(len(true_labels), size=len(true_labels), replace=True)
    bootstrap_true_labels = true_labels[bootstrap_indices]
    bootstrap_predicted_labels = predicted_labels[bootstrap_indices]
    bootstrap_predicted_probs = predicted_probs[bootstrap_indices]

    accuracy_bootstrap.append(accuracy_score(bootstrap_true_labels, bootstrap_predicted_labels))
    precision_bootstrap.append(precision_score(bootstrap_true_labels, bootstrap_predicted_labels))
    recall_bootstrap.append(recall_score(bootstrap_true_labels, bootstrap_predicted_labels))
    f1_bootstrap.append(f1_score(bootstrap_true_labels, bootstrap_predicted_labels))
    roc_auc_bootstrap.append(roc_auc_score(bootstrap_true_labels, bootstrap_predicted_probs))

accuracy_ci_bootstrap = np.percentile(accuracy_bootstrap, [2.5, 97.5])
precision_ci_bootstrap = np.percentile(precision_bootstrap, [2.5, 97.5])
recall_ci_bootstrap = np.percentile(recall_bootstrap, [2.5, 97.5])
f1_ci_bootstrap = np.percentile(f1_bootstrap, [2.5, 97.5])
roc_auc_ci_bootstrap = np.percentile(roc_auc_bootstrap, [2.5, 97.5])

print(f"95% CI for Accuracy (Bootstrap): {accuracy_ci_bootstrap}")
print(f"95% CI for Precision (Bootstrap): {precision_ci_bootstrap}")
print(f"95% CI for Recall (Bootstrap): {recall_ci_bootstrap}")
print(f"95% CI for F1 Score (Bootstrap): {f1_ci_bootstrap}")
print(f"95% CI for ROC AUC (Bootstrap): {roc_auc_ci_bootstrap}")

######################
# Confusion Matrix   #
######################

conf_matrix = confusion_matrix(true_labels, predicted_labels)
row_sums = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_percent_row = conf_matrix / row_sums * 100
annotations = np.array([
    f"{conf_matrix[i, j]:d}\n({conf_matrix_percent_row[i, j]:.2f}%)"
    for i in range(conf_matrix.shape[0])
    for j in range(conf_matrix.shape[1])
]).reshape(conf_matrix.shape)

colors = sns.diverging_palette(80, 5, s=70, l=80, as_cmap=True)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=annotations, fmt="", cmap=colors, cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])
plt.yticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])
plt.savefig("confusion_matrix.pdf", dpi=1000)
plt.show()

######################
# ROC Curve          #
######################

fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc_val = auc(fpr, tpr)

sns.set_style("white")
sns.set_palette(["#e090b5", "#e6cd73"])

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label="ROC curve (area = %0.4f)" % roc_auc_val)
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.pdf", dpi=1000)
plt.show()
