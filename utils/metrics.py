import torch
import torchmetrics
from pathlib import Path

def get_metrics(n_class, device):
    precision_metric = torchmetrics.Precision(
        task="multiclass",
        num_classes=n_class,
        average="macro"
    ).to(device)

    recall_metric = torchmetrics.Recall(
        task="multiclass",
        num_classes=n_class,
        average="macro"
    ).to(device)

    f1_metric = torchmetrics.F1Score(
        task="multiclass",
        num_classes=n_class,
        average="macro"
    ).to(device)

    return precision_metric, recall_metric, f1_metric

def save_model(model, path, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": class_names
    }
    torch.save(checkpoint, path)

def print_epoch_result(epoch, epochs, train_loss, train_acc, val_loss, val_acc, metrics):
    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"  [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"  [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

    if metrics:
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)

        print(
            f"  [Metrics] Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | F1: {f1:.4f}"
        )

    print("-" * 40)