####################################
# Model evaluation (standalone)    #
####################################

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved ResNet50-based binary classifier checkpoint.")
    p.add_argument("--data-root", type=Path, default=Path("dataset_splits"), help="ImageFolder root containing test/.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to model_best.pth")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("runs") / "eval")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    ROC-AUC via rank statistic (Mann–Whitney U). Returns 0.0 if undefined.
    """
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n = len(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
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
    return float((sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def roc_curve_points(y_true: list[int], y_score: list[float]) -> tuple[list[float], list[float]]:
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
    return [0.0] + fprs + [1.0], [0.0] + tprs + [1.0]


def trapz_auc(x: list[float], y: list[float]) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return float(area)


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    test_dir = args.data_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {args.checkpoint}")

    eval_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            y_prob.extend(outputs.view(-1).detach().cpu().tolist())
            y_true.extend([int(x) for x in labels.view(-1).tolist()])

    y_pred = [1 if p > 0.5 else 0 for p in y_prob]

    acc = accuracy(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    auc_val = roc_auc(y_true, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc_val:.4f}")

    # Bootstrap CIs (optional but useful for papers)
    n_iterations = 1000
    acc_bs: list[float] = []
    prec_bs: list[float] = []
    rec_bs: list[float] = []
    f1_bs: list[float] = []
    auc_bs: list[float] = []
    for _ in range(n_iterations):
        idxs = [random.randrange(len(y_true)) for _ in range(len(y_true))]
        bt_true = [y_true[i] for i in idxs]
        bt_prob = [y_prob[i] for i in idxs]
        bt_pred = [1 if p > 0.5 else 0 for p in bt_prob]
        acc_bs.append(accuracy(bt_true, bt_pred))
        p, r, f = precision_recall_f1(bt_true, bt_pred)
        prec_bs.append(p)
        rec_bs.append(r)
        f1_bs.append(f)
        auc_bs.append(roc_auc(bt_true, bt_prob))

    print(f"95% CI Accuracy: [{percentile(acc_bs, 2.5):.4f}, {percentile(acc_bs, 97.5):.4f}]")
    print(f"95% CI Precision: [{percentile(prec_bs, 2.5):.4f}, {percentile(prec_bs, 97.5):.4f}]")
    print(f"95% CI Recall: [{percentile(rec_bs, 2.5):.4f}, {percentile(rec_bs, 97.5):.4f}]")
    print(f"95% CI F1: [{percentile(f1_bs, 2.5):.4f}, {percentile(f1_bs, 97.5):.4f}]")
    print(f"95% CI ROC AUC: [{percentile(auc_bs, 2.5):.4f}, {percentile(auc_bs, 97.5):.4f}]")

    # Confusion matrix plot
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    conf = [[tn, fp], [fn, tp]]
    plt.figure(figsize=(6, 6))
    if sns is not None:
        colors = sns.diverging_palette(80, 5, s=70, l=80, as_cmap=True)
        sns.heatmap(conf, annot=True, fmt="d", cmap=colors, cbar=False)
    else:
        plt.imshow(conf, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"])
    plt.yticks(ticks=[0, 1], labels=["Benign", "Malignant"])
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.pdf", dpi=600)
    plt.close()

    # ROC curve plot
    fpr, tpr = roc_curve_points(y_true, y_prob)
    roc_area = trapz_auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (area = {roc_area:.4f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curve.pdf", dpi=600)
    plt.close()

    print(f"Saved figures under: {args.output_dir}")


if __name__ == "__main__":
    main()
