from __future__ import annotations

"""
ConvNeXt-Tiny staged training (revival CLI).

Expects a torchvision ImageFolder layout:
  data_root/
    train/{BENIGN,MALIGNANT}/...
    val/{BENIGN,MALIGNANT}/...
    test/{BENIGN,MALIGNANT}/...

This script intentionally avoids numpy/sklearn to keep it runnable in minimal environments.
"""

import argparse
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights

import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None


@dataclass(frozen=True)
class Config:
    data_root: Path
    output_dir: Path
    image_size: int = 384
    batch_size: int = 64
    head_epochs: int = 20
    finetune_epochs: int = 40
    lr_head: float = 1e-3
    lr_fine: float = 1e-4
    step_size: int = 7
    gamma: float = 0.1
    subset_size: int = 0  # 0 = full
    num_workers: int = 0
    seed: int = 42
    early_stop_patience: int = 7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ConvNeXt-Tiny (staged) on ImageFolder dataset_splits.")
    p.add_argument("--data-root", type=Path, default=Path("dataset_splits"))
    p.add_argument("--output-dir", type=Path, default=Path("runs") / "convnext_stage")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--head-epochs", type=int, default=20)
    p.add_argument("--finetune-epochs", type=int, default=40)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-fine", type=float, default=1e-4)
    p.add_argument("--step-size", type=int, default=7)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--subset-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=7)
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_metrics_row(path: Path, row: dict[str, str]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


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


class ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        self.feature_extractor = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(dim=[2, 3])
        return self.head(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        focal_factor = torch.where(targets == 1, (1 - probs) ** self.gamma, probs ** self.gamma)
        loss = self.alpha * focal_factor * base_loss
        return loss.mean()


def compute_pos_weight(counts: Counter[int]) -> torch.Tensor:
    neg = counts.get(0, 0)
    pos = counts.get(1, 0)
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(train_mode):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs).view(-1)
            loss = criterion(logits, labels.float())

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))

    acc = total_correct / total_samples if total_samples else 0.0
    avg_loss = total_loss / max(1, len(loader))
    return acc, avg_loss


def evaluate_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).view(-1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            y_prob.extend([float(p) for p in probs])
            y_true.extend([int(x) for x in labels.view(-1).tolist()])
    return y_true, y_prob


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        lr_head=args.lr_head,
        lr_fine=args.lr_fine,
        step_size=args.step_size,
        gamma=args.gamma,
        subset_size=args.subset_size,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    metrics_path = cfg.output_dir / "training_metrics_stage.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    train_dir = cfg.data_root / "train"
    val_dir = cfg.data_root / "val"
    test_dir = cfg.data_root / "test"
    for p in (train_dir, val_dir, test_dir):
        if not p.exists():
            raise FileNotFoundError(f"Required directory missing: {p}")

    transform_train = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=transform_train)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=transform_eval)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=transform_eval)

    if cfg.subset_size and cfg.subset_size > 0:
        train_dataset = Subset(train_dataset, range(min(cfg.subset_size, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(cfg.subset_size, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(cfg.subset_size, len(test_dataset))))

    # Build class-balanced sampler (counts from labels)
    train_targets: list[int] = []
    for _, labels in DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0):
        train_targets.extend([int(x) for x in labels.view(-1).tolist()])
    class_counts: Counter[int] = Counter(train_targets)
    total = sum(class_counts.values()) if class_counts else 1
    class_weights = {cls: total / cnt for cls, cnt in class_counts.items() if cnt > 0}
    sample_weights = torch.tensor([class_weights[int(lbl)] for lbl in train_targets], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtTiny().to(device)
    pos_weight = compute_pos_weight(class_counts).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=1.0, pos_weight=pos_weight)

    # Stage 1: freeze backbone
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.head.parameters(), lr=cfg.lr_head)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    train_losses_stage1: list[float] = []
    val_losses_stage1: list[float] = []
    for epoch in range(cfg.head_epochs):
        train_acc, train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_acc, val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        train_losses_stage1.append(train_loss)
        val_losses_stage1.append(val_loss)
        print(
            f"Stage1 {epoch+1}/{cfg.head_epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        write_metrics_row(
            metrics_path,
            dict(
                stage="1",
                epoch=str(epoch + 1),
                train_accuracy=f"{train_acc:.6f}",
                val_accuracy=f"{val_acc:.6f}",
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
            ),
        )

    # Stage 2: unfreeze backbone
    for param in model.feature_extractor.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_fine, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.finetune_epochs)

    best_val = 0.0
    best_ckpt = cfg.output_dir / "model_best_stage.pth"
    epochs_no_improve = 0
    train_losses_stage2: list[float] = []
    val_losses_stage2: list[float] = []

    for epoch in range(cfg.finetune_epochs):
        train_acc, train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_acc, val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        train_losses_stage2.append(train_loss)
        val_losses_stage2.append(val_loss)
        print(
            f"Stage2 {epoch+1}/{cfg.finetune_epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        write_metrics_row(
            metrics_path,
            dict(
                stage="2",
                epoch=str(epoch + 1),
                train_accuracy=f"{train_acc:.6f}",
                val_accuracy=f"{val_acc:.6f}",
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
            ),
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.early_stop_patience:
            print("Early stopping triggered.")
            break

    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint saved at {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Evaluate on test set
    y_true, y_prob = evaluate_probs(model, test_loader, device)
    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    acc = accuracy(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    auc_val = roc_auc(y_true, y_prob)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    print(
        "Test metrics: "
        f"acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} "
        f"roc_auc={auc_val:.4f} specificity={specificity:.4f}"
    )

    # Confusion matrix plot
    conf = [[tn, fp], [fn, tp]]
    plt.figure(figsize=(6, 6))
    if sns is not None:
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
    else:
        plt.imshow(conf, cmap="Blues")
    plt.title("ConvNeXt Staged Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Benign", "Malignant"])
    plt.yticks([0, 1], ["Benign", "Malignant"])
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "confusion_matrix_stage.pdf", dpi=600)
    plt.close()

    # ROC curve plot
    fpr, tpr = roc_curve_points(y_true, y_prob)
    roc_area = trapz_auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_area:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ConvNeXt Staged ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "roc_curve_stage.pdf", dpi=600)
    plt.close()

    # Loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses_stage1) + 1), train_losses_stage1, label="Stage1 Train Loss")
    plt.plot(range(1, len(val_losses_stage1) + 1), val_losses_stage1, label="Stage1 Val Loss")
    plt.plot(range(1, len(train_losses_stage2) + 1), train_losses_stage2, label="Stage2 Train Loss")
    plt.plot(range(1, len(val_losses_stage2) + 1), val_losses_stage2, label="Stage2 Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "loss_curves.pdf", dpi=600)
    plt.close()

    print(f"Run artifacts saved under: {cfg.output_dir}")


if __name__ == "__main__":
    main()


