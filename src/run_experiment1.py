# run_experiment1.py
# Experiment 1 — Baseline CNN
# Guarda automáticamente resultados, gráficas, macro-F1, matriz de confusión,
# logs permanentes y tablas requeridas para el Experimento 1.

import csv
import json
import time
import random
import platform
from pathlib import Path
from datetime import datetime
from collections import Counter
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from baseline_cnn import BaselineCNN


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

EXPERIMENT_NAME = "Exp 1"
MODEL_NAME = "BaselineCNN"

# Opciones para la tabla obligatoria:
# "Group", "ODD", "EVEN", "ODD+EVEN", "Public"
DATASET_TABLE_NAME = "Public"

# Nombre para carpetas de salida.
# Ejemplos: "OWN", "G00", "Kaggle", "ODD", "EVEN", "ODD_EVEN"
DATASET_NAME = "Kaggle"

# Carpeta real del dataset.
# Debe tener:
# dataset/
#   train/
#   val/
#   test/
DATASET_FOLDER_NAME = "Kaggle"

# Hiperparámetros Experimento 1
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.01
DROPOUT = 0.5
NUM_CLASSES = 6
NUM_WORKERS = 4
SEED = 42

# Si quieres que se pare si no detecta CUDA
REQUIRE_CUDA = True

# Guardar ejemplos mal clasificados
SAVE_MISCLASSIFIED = True
MAX_MISCLASSIFIED_IMAGES = 24


# =============================================================================
# RUTAS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / DATASET_FOLDER_NAME
OUTPUT_DIR = BASE_DIR / f"output_{DATASET_NAME}"

TABLES_DIR = BASE_DIR / "experiment1_results_tables"
GLOBAL_LOG_PATH = TABLES_DIR / "experiment1_global_log.csv"
TABLE1_PATH = TABLES_DIR / "table1_main_performance_exp1.csv"
TABLE4_PATH = TABLES_DIR / "table4_computational_cost_exp1.csv"

DATASET_LOG_PATH = OUTPUT_DIR / "experiment1_dataset_log.csv"
TEXT_LOG_PATH = OUTPUT_DIR / "experiment1_log.txt"

# Se crea dentro de main(), no aquí.
RUN_DIR = None


# =============================================================================
# CUDA
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if REQUIRE_CUDA and DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA no está disponible. Revisa que PyTorch con CUDA esté instalado "
        "y que estés usando el intérprete correcto."
    )

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# =============================================================================
# REPRODUCIBILIDAD
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TRANSFORMS
# =============================================================================

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


# =============================================================================
# DATASET
# =============================================================================

def check_dataset_structure(data_dir: Path):
    for split in ["train", "val", "test"]:
        split_path = data_dir / split

        if not split_path.exists():
            raise FileNotFoundError(f"No existe la carpeta: {split_path}")

        if not split_path.is_dir():
            raise NotADirectoryError(f"No es una carpeta válida: {split_path}")


def print_class_distribution(dataset, split_name: str):
    counts = Counter(dataset.targets)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    print(f"\n{split_name} distribution:")
    for idx in range(len(idx_to_class)):
        class_name = idx_to_class[idx]
        count = counts[idx]
        print(f"  {class_name:12s}: {count:5d}")


def get_dataloaders(data_dir: Path):
    check_dataset_structure(data_dir)

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

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError("Las clases de train y val no coinciden.")

    if train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("Las clases de train y test no coinciden.")

    print(f"Dataset path: {data_dir}")
    print(f"Classes: {train_ds.classes}")
    print(f"Class mapping: {train_ds.class_to_idx}")

    print(f"\nTrain images     : {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Test images      : {len(test_ds)}")

    print_class_distribution(train_ds, "Train")
    print_class_distribution(val_ds, "Validation")
    print_class_distribution(test_ds, "Test")
    print()

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, train_ds.classes


# =============================================================================
# MODELO
# =============================================================================

def create_model():
    model = BaselineCNN(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT,
    )

    return model.to(DEVICE)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# TRAIN / EVALUATE
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad() #limpiar gradientes

        outputs = model(images) #fp
        loss = criterion(outputs, labels) #calculamos errror

        loss.backward()
        optimizer.step() #actualizamos pesos

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_accuracy, epoch_macro_f1


def evaluate(model, loader, criterion, return_predictions=False):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_paths = []

    dataset = loader.dataset
    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            labels_cpu = labels.detach().cpu().numpy()
            predictions_cpu = predictions.detach().cpu().numpy()

            all_labels.extend(labels_cpu)
            all_predictions.extend(predictions_cpu)

            if return_predictions:
                for _ in range(batch_size):
                    if sample_index < len(dataset.samples):
                        all_paths.append(dataset.samples[sample_index][0])
                    sample_index += 1

    epoch_loss = total_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    if return_predictions:
        return epoch_loss, epoch_accuracy, epoch_macro_f1, all_labels, all_predictions, all_paths

    return epoch_loss, epoch_accuracy, epoch_macro_f1


# =============================================================================
# FIGURAS
# =============================================================================

def save_training_curves(history):
    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.title(f"{EXPERIMENT_NAME} - {DATASET_NAME} - Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.title(f"{EXPERIMENT_NAME} - {DATASET_NAME} - Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "accuracy_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.title("Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.title("Accuracy Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RUN_DIR / "training_curves_combined.png", dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title(f"{EXPERIMENT_NAME} - {DATASET_NAME} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def save_misclassified_examples(y_true, y_pred, paths, class_names):
    if not SAVE_MISCLASSIFIED:
        return

    misclassified = []

    for true_label, predicted_label, path in zip(y_true, y_pred, paths):
        if true_label != predicted_label:
            misclassified.append((true_label, predicted_label, path))

    csv_path = RUN_DIR / "misclassified_examples.csv"

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_label", "predicted_label"])

        for true_label, predicted_label, path in misclassified:
            writer.writerow([
                path,
                class_names[true_label],
                class_names[predicted_label],
            ])

    if len(misclassified) == 0:
        return

    examples = misclassified[:MAX_MISCLASSIFIED_IMAGES]
    cols = 4
    rows = int(np.ceil(len(examples) / cols))

    plt.figure(figsize=(cols * 4, rows * 4))

    for i, (true_label, predicted_label, path) in enumerate(examples, start=1):
        img = plt.imread(path)

        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(RUN_DIR / "misclassified_examples.png", dpi=150)
    plt.close()


# =============================================================================
# GUARDADO DE RESULTADOS
# =============================================================================

def save_training_history(history):
    csv_path = RUN_DIR / "training_history.csv"

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(history.keys())

        rows = zip(*history.values())
        writer.writerows(rows)


def save_classification_report(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    with open(RUN_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    print("\nClassification report:")
    print(report)


def get_hardware_description():
    if torch.cuda.is_available():
        return (
            f"{torch.cuda.get_device_name(0)} | "
            f"CUDA {torch.version.cuda} | "
            f"CPU {platform.processor()}"
        )

    return f"CPU only | {platform.processor()}"


def append_text_log(text):
    with open(TEXT_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def save_summary(summary):
    with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)


def append_csv_row(csv_path, row):
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def update_table1_exp1(summary):
    columns = ["Experiment", "Group", "ODD", "EVEN", "ODD+EVEN", "Public"]

    if TABLE1_PATH.exists():
        rows = []
        with open(TABLE1_PATH, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(row)
    else:
        rows = []

    found = False

    for row in rows:
        if row["Experiment"] == "Exp 1":
            row[DATASET_TABLE_NAME] = f"{summary['test_macro_f1']:.4f}"
            found = True

    if not found:
        row = {col: "" for col in columns}
        row["Experiment"] = "Exp 1"
        row[DATASET_TABLE_NAME] = f"{summary['test_macro_f1']:.4f}"
        rows.append(row)

    with open(TABLE1_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def update_table4_exp1(summary):
    row = {
        "Model": MODEL_NAME,
        "Training Time": f"{summary['training_time_seconds']:.2f} s",
        "Hardware": summary["hardware"],
        "Batch Size": BATCH_SIZE,
        "Experiment": "Exp 1",
        "Dataset": DATASET_NAME,
    }

    append_csv_row(TABLE4_PATH, row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global RUN_DIR

    # Las carpetas se crean SOLO aquí.
    # Así los workers de Windows no crean carpetas vacías.
    OUTPUT_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = OUTPUT_DIR / f"run_{run_id}"
    RUN_DIR.mkdir(exist_ok=True)

    set_seed(SEED)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Dataset    : {DATASET_NAME}")
    print(f"Data path  : {DATA_DIR}")
    print(f"Output dir : {RUN_DIR}")
    print(f"Device     : {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"CUDA       : {torch.version.cuda}")

    print("=" * 80)

    append_text_log("\n" + "=" * 80)
    append_text_log(f"RUN START: {timestamp}")
    append_text_log(f"Experiment: {EXPERIMENT_NAME}")
    append_text_log(f"Dataset: {DATASET_NAME}")
    append_text_log(f"Data path: {DATA_DIR}")
    append_text_log(f"Output dir: {RUN_DIR}")
    append_text_log(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR)

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR)

    num_parameters = count_parameters(model)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "epoch_time_seconds": [],
    }

    print(f"\nModel parameters: {num_parameters:,}")
    print(f"Optimizer       : SGD")
    print(f"Learning rate   : {LR}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Dropout         : {DROPOUT}")
    print(f"Workers         : {NUM_WORKERS}\n")

    header = (
        f"{'Epoch':>6} "
        f"{'Train Loss':>12} "
        f"{'Train Acc':>10} "
        f"{'Train F1':>10} "
        f"{'Val Loss':>10} "
        f"{'Val Acc':>10} "
        f"{'Val F1':>10} "
        f"{'Time':>8}"
    )

    print(header)
    print("-" * len(header))
    append_text_log(header)
    append_text_log("-" * len(header))

    best_val_macro_f1 = -1.0
    best_val_accuracy = -1.0
    best_epoch = -1

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_accuracy, train_macro_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_accuracy, val_macro_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            return_predictions=False,
        )

        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_macro_f1"].append(val_macro_f1)
        history["epoch_time_seconds"].append(epoch_time)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), RUN_DIR / "best_model_weights.pth")

        line = (
            f"{epoch:>6} "
            f"{train_loss:>12.4f} "
            f"{train_accuracy:>9.2%} "
            f"{train_macro_f1:>10.4f} "
            f"{val_loss:>10.4f} "
            f"{val_accuracy:>9.2%} "
            f"{val_macro_f1:>10.4f} "
            f"{epoch_time:>7.1f}s"
        )

        print(line)
        append_text_log(line)

    training_time = time.time() - total_start

    print("\nFinal evaluation on TEST set")

    test_loss, test_accuracy, test_macro_f1, y_true, y_pred, test_paths = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        return_predictions=True,
    )

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.2%}")
    print(f"Test Macro-F1 : {test_macro_f1:.4f}")

    save_training_history(history)
    save_training_curves(history)
    save_confusion_matrix(y_true, y_pred, class_names)
    save_misclassified_examples(y_true, y_pred, test_paths, class_names)
    save_classification_report(y_true, y_pred, class_names)

    torch.save(model.state_dict(), RUN_DIR / "final_model_weights.pth")

    summary = {
        "timestamp": timestamp,
        "experiment": EXPERIMENT_NAME,
        "dataset": DATASET_NAME,
        "dataset_table_name": DATASET_TABLE_NAME,
        "dataset_path": str(DATA_DIR),
        "output_dir": str(RUN_DIR),
        "model": MODEL_NAME,
        "num_parameters": num_parameters,
        "optimizer": "SGD",
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "dropout": DROPOUT,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "device": str(DEVICE),
        "hardware": get_hardware_description(),
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "classes": class_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "training_time_seconds": training_time,
    }

    save_summary(summary)

    log_row = {
        "timestamp": timestamp,
        "experiment": EXPERIMENT_NAME,
        "dataset": DATASET_NAME,
        "dataset_table_name": DATASET_TABLE_NAME,
        "model": MODEL_NAME,
        "optimizer": "SGD",
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "dropout": DROPOUT,
        "num_workers": NUM_WORKERS,
        "device": str(DEVICE),
        "hardware": get_hardware_description(),
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "training_time_seconds": training_time,
        "output_dir": str(RUN_DIR),
    }

    append_csv_row(GLOBAL_LOG_PATH, log_row)
    append_csv_row(DATASET_LOG_PATH, log_row)

    update_table1_exp1(summary)
    update_table4_exp1(summary)

    append_text_log(f"Test Loss: {test_loss:.4f}")
    append_text_log(f"Test Accuracy: {test_accuracy:.4f}")
    append_text_log(f"Test Macro-F1: {test_macro_f1:.4f}")
    append_text_log(f"Best Val Macro-F1: {best_val_macro_f1:.4f}")
    append_text_log(f"Best epoch: {best_epoch}")
    append_text_log(f"Training time: {training_time:.2f} seconds")
    append_text_log(f"RUN END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    append_text_log("=" * 80)

    print("\nSaved in:")
    print(RUN_DIR)

    print("\nUpdated tables/logs:")
    print(GLOBAL_LOG_PATH)
    print(TABLE1_PATH)
    print(TABLE4_PATH)

    print("\nDone.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()