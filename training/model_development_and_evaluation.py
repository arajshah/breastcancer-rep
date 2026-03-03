####################################
# Model development and evaluation #
####################################

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import ResNet50_Weights

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

################################
# CLI / configuration          #
################################

@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    output_dir: Path
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-6
    seed: int = 42
    num_workers: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + evaluate a ResNet50-based binary classifier (CBIS-DDSM Mass).")
    p.add_argument("--data-root", type=Path, default=Path("dataset_splits"), help="ImageFolder root with train/val/test.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory (checkpoints/metrics/figures). Default: runs/resnet50_YYYYmmdd_HHMMSS",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sigmoid(x: float) -> float:
    # numerically stable sigmoid for scalar
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tn, fp, fn, tp


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred, strict=True))
    return correct / len(y_true) if y_true else 0.0


def roc_auc(y_true: list[int], y_score: list[float]) -> float:
    """
    Compute ROC-AUC using rank statistic (equivalent to Mann–Whitney U).
    Returns 0.0 if undefined (all positives or all negatives).
    """
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n = len(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # average ranks for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(rank for rank, (_, y) in zip(ranks, pairs, strict=True) if y == 1)
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def roc_curve_points(y_true: list[int], y_score: list[float]) -> tuple[list[float], list[float]]:
    """
    Simple ROC curve (FPR/TPR) computed by sweeping thresholds over sorted unique scores.
    """
    # thresholds descending
    thresholds = sorted(set(y_score), reverse=True)
    fprs: list[float] = []
    tprs: list[float] = []
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0]
    for thr in thresholds:
        y_pred = [1 if s >= thr else 0 for s in y_score]
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fprs.append(fpr)
        tprs.append(tpr)
    # Ensure endpoints
    fprs = [0.0] + fprs + [1.0]
    tprs = [0.0] + tprs + [1.0]
    return fprs, tprs


def trapz_auc(x: list[float], y: list[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return float(area)


def write_metrics_row(path: Path, row: dict[str, str]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"resnet50_{ts}"
    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=out_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    checkpoint_path = cfg.output_dir / "model_best.pth"
    metrics_path = cfg.output_dir / "training_metrics.csv"

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dir = cfg.data_root / "train"
val_dir = cfg.data_root / "val"
test_dir = cfg.data_root / "test"
for p in (train_dir, val_dir, test_dir):
    if not p.exists():
        raise FileNotFoundError(f"Required directory missing: {p}")

train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_transforms)
test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transforms)

print(f"Classes: {train_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

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
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_accuracy = 0.0

for epoch in range(cfg.epochs):
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
        f"Epoch [{epoch+1}/{cfg.epochs}], "
        f"Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    )
    write_metrics_row(
        metrics_path,
        dict(
            epoch=str(epoch + 1),
            train_accuracy=f"{train_accuracy:.6f}",
            val_accuracy=f"{val_accuracy:.6f}",
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
        ),
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), checkpoint_path)
        print(f"New best checkpoint saved: {checkpoint_path} (val_acc={val_accuracy:.4f})")

####################
# Model evaluation #
####################

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}. Train the model first.")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

true_labels = []
predicted_probs = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = model(test_images)
        # outputs: (B,1) -> list[float]
        predicted_probs.extend(outputs.view(-1).detach().cpu().tolist())
        true_labels.extend(test_labels.view(-1).detach().cpu().tolist())

true_labels_int = [int(x) for x in true_labels]
predicted_labels_int = [1 if p > 0.5 else 0 for p in predicted_probs]

acc = accuracy(true_labels_int, predicted_labels_int)
prec, rec, f1 = precision_recall_f1(true_labels_int, predicted_labels_int)
auc_val = roc_auc(true_labels_int, predicted_probs)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"ROC AUC: {auc_val:.4f}")
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
    idxs = [random.randrange(len(true_labels_int)) for _ in range(len(true_labels_int))]
    bt_true = [true_labels_int[i] for i in idxs]
    bt_probs = [predicted_probs[i] for i in idxs]
    bt_pred = [1 if p > 0.5 else 0 for p in bt_probs]
    accuracy_bootstrap.append(accuracy(bt_true, bt_pred))
    p, r, f = precision_recall_f1(bt_true, bt_pred)
    precision_bootstrap.append(p)
    recall_bootstrap.append(r)
    f1_bootstrap.append(f)
    roc_auc_bootstrap.append(roc_auc(bt_true, bt_probs))

def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))

print(f"95% CI Accuracy (Bootstrap): [{percentile(accuracy_bootstrap, 2.5):.4f}, {percentile(accuracy_bootstrap, 97.5):.4f}]")
print(f"95% CI Precision (Bootstrap): [{percentile(precision_bootstrap, 2.5):.4f}, {percentile(precision_bootstrap, 97.5):.4f}]")
print(f"95% CI Recall (Bootstrap): [{percentile(recall_bootstrap, 2.5):.4f}, {percentile(recall_bootstrap, 97.5):.4f}]")
print(f"95% CI F1 (Bootstrap): [{percentile(f1_bootstrap, 2.5):.4f}, {percentile(f1_bootstrap, 97.5):.4f}]")
print(f"95% CI ROC AUC (Bootstrap): [{percentile(roc_auc_bootstrap, 2.5):.4f}, {percentile(roc_auc_bootstrap, 97.5):.4f}]")

######################
# Confusion Matrix   #
######################

tn, fp, fn, tp = confusion_counts(true_labels_int, predicted_labels_int)
conf_matrix = [[tn, fp], [fn, tp]]
row_sums = [sum(conf_matrix[0]), sum(conf_matrix[1])]
conf_pct = [
    [conf_matrix[0][0] / row_sums[0] * 100 if row_sums[0] else 0.0, conf_matrix[0][1] / row_sums[0] * 100 if row_sums[0] else 0.0],
    [conf_matrix[1][0] / row_sums[1] * 100 if row_sums[1] else 0.0, conf_matrix[1][1] / row_sums[1] * 100 if row_sums[1] else 0.0],
]
plt.figure(figsize=(6, 6))
if sns is not None:
    colors = sns.diverging_palette(80, 5, s=70, l=80, as_cmap=True)
    annot = [
        [f"{conf_matrix[0][0]}\n({conf_pct[0][0]:.2f}%)", f"{conf_matrix[0][1]}\n({conf_pct[0][1]:.2f}%)"],
        [f"{conf_matrix[1][0]}\n({conf_pct[1][0]:.2f}%)", f"{conf_matrix[1][1]}\n({conf_pct[1][1]:.2f}%)"],
    ]
    sns.heatmap(conf_matrix, annot=annot, fmt="", cmap=colors, cbar=False)
else:
    plt.imshow(conf_matrix, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"])
plt.yticks(ticks=[0, 1], labels=["Benign", "Malignant"])
plt.tight_layout()
plt.savefig(cfg.output_dir / "confusion_matrix.pdf", dpi=600)
plt.close()

######################
# ROC Curve          #
######################

fpr, tpr = roc_curve_points(true_labels_int, predicted_probs)
roc_auc_val = trapz_auc(fpr, tpr)

if sns is not None: sns.set_style("white"); sns.set_palette(["#e090b5", "#e6cd73"])

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label="ROC curve (area = %0.4f)" % roc_auc_val)
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(cfg.output_dir / "roc_curve.pdf", dpi=600)
plt.close()

print(f"Run artifacts saved under: {cfg.output_dir}")


if __name__ == "__main__":
    main()
