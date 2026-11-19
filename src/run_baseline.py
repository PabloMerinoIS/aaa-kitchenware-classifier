# Experiment 1 – Baseline CNN Training

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, f1_score

from baseline_cnn import BaselineCNN


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "G00_dataset_split"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.01
NUM_CLASSES = 6
DROPOUT = 0.5
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ── Dataloaders ───────────────────────────────────────────────────────────────
def get_dataloaders(data_dir):
    train_ds = datasets.ImageFolder(
        root=data_dir / "train",
        transform=train_transforms,
    )

    val_ds = datasets.ImageFolder(
        root=data_dir / "val",
        transform=eval_transforms,
    )

    test_ds = datasets.ImageFolder(
        root=data_dir / "test",
        transform=eval_transforms,
    )

    print(f"Classes: {train_ds.classes}")
    print(f"Train images     : {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Test images      : {len(test_ds)}\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_accuracy = correct / len(loader.dataset)

    return epoch_loss, epoch_accuracy


# ── Validation / Evaluation without predictions ───────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_accuracy = correct / len(loader.dataset)

    return epoch_loss, epoch_accuracy


# ── Final test evaluation with predictions for Macro-F1 ───────────────────────
def evaluate_with_predictions(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_accuracy = correct / len(loader.dataset)

    return epoch_loss, epoch_accuracy, all_labels, all_predictions


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")

    plt.title("Train Loss vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "train_vs_val_loss.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Loss comparison graph saved: {save_path}")


def plot_validation_accuracy(val_accuracies):
    epochs = range(1, len(val_accuracies) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")

    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "validation_accuracy.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Validation accuracy graph saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print()

    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR)

    model = BaselineCNN(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10} {'Time':>8}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
        )

        elapsed = time.time() - start_time

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"{epoch:>6} "
            f"{train_loss:>12.4f} "
            f"{train_accuracy:>9.2%} "
            f"{val_loss:>10.4f} "
            f"{val_accuracy:>9.2%} "
            f"{elapsed:>7.1f}s"
        )

    print("\nFinal evaluation on TEST set")

    test_loss, test_accuracy, y_true, y_pred = evaluate_with_predictions(
        model=model,
        loader=test_loader,
        criterion=criterion,
    )

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.2%}")
    print(f"Macro-F1      : {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=test_loader.dataset.classes,
            digits=4,
            zero_division=0,
        )
    )

    plot_loss_curves(train_losses, val_losses)
    plot_validation_accuracy(val_accuracies)

    weights_path = OUTPUT_DIR / "baseline_cnn_weights.pth"
    torch.save(model.state_dict(), weights_path)

    print(f"Model weights saved: {weights_path}")


if __name__ == "__main__":
    main()