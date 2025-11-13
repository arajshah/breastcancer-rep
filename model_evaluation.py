####################################
# Model evaluation (standalone)    #
####################################

from __future__ import annotations

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

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
import matplotlib.pyplot as plt

################################
# Paths and transforms         #
################################

DATA_ROOT = Path("/Users/saujanyathapaliya/Documents/breastcancer/processed_png/dataset_splits")
TEST_DIR = DATA_ROOT / "test"
CHECKPOINT_PATH = Path("/Users/saujanyathapaliya/Documents/breastcancer/processed_png/checkpoints/model_best.pth")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Missing test directory: {TEST_DIR}")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Missing checkpoint file: {CHECKPOINT_PATH}")

eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

###################
# Model definition#
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

##########################
# Evaluation             #
##########################

true_labels = []
predicted_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted_probs.extend(outputs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

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
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

##########################
# Confidence Intervals   #
##########################

n_iterations = 1000
accuracy_bootstrap = []
precision_bootstrap = []
recall_bootstrap = []
f1_bootstrap = []
roc_auc_bootstrap = []

for _ in range(n_iterations):
    bootstrap_idx = np.random.choice(len(true_labels), size=len(true_labels), replace=True)
    boot_true = true_labels[bootstrap_idx]
    boot_pred = predicted_labels[bootstrap_idx]
    boot_probs = predicted_probs[bootstrap_idx]

    accuracy_bootstrap.append(accuracy_score(boot_true, boot_pred))
    precision_bootstrap.append(precision_score(boot_true, boot_pred))
    recall_bootstrap.append(recall_score(boot_true, boot_pred))
    f1_bootstrap.append(f1_score(boot_true, boot_pred))
    roc_auc_bootstrap.append(roc_auc_score(boot_true, boot_probs))

print(f"95% CI Accuracy: {np.percentile(accuracy_bootstrap, [2.5, 97.5])}")
print(f"95% CI Precision: {np.percentile(precision_bootstrap, [2.5, 97.5])}")
print(f"95% CI Recall: {np.percentile(recall_bootstrap, [2.5, 97.5])}")
print(f"95% CI F1: {np.percentile(f1_bootstrap, [2.5, 97.5])}")
print(f"95% CI ROC AUC: {np.percentile(roc_auc_bootstrap, [2.5, 97.5])}")

##########################
# Confusion Matrix       #
##########################

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

##########################
# ROC Curve              #
##########################

fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
roc_auc_val = auc(fpr, tpr)

sns.set_style("white")

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
