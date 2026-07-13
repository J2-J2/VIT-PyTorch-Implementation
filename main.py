import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from models import *
import argparse
from data import get_loader
from models import VIT
from utils import train_one_epoch, validate, get_metrics, save_model, print_epoch_result
import json
from datetime import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser(description="vit_base_pytorch")

    parser.add_argument("--config", type=str, default="configs/vit_custom.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    return parser.parse_args()




def main():
    # os.environ["PYTHONBREAKPOINT"] = "pdb.set_trace"
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
        
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
        
    if args.lr is not None:
        config['train']['lr'] = args.lr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(config["save"]["checkpoint_dir"]) / current_time
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path.joinpath(config["save"]["best_model_name"])

    train_loader = get_loader(config, mode="train")
    val_loader = get_loader(config, mode="val")
    class_names = train_loader.dataset.class_names
    # breakpoint()

    print("\n" + "="*30 + " TRAINING CONFIGURATION " + "="*30)
    print(json.dumps(config, indent=4, ensure_ascii=False))
    print("="*88 + "\n")

    model_cfg = config["model"]
    data_cfg = config["data"]

    model = VIT(
        img_size    = data_cfg["image_size"],
        patch_size  = data_cfg["patch_size"],
        n_layer     = data_cfg["n_layers"],
        input_dim   = data_cfg["input_dim"],
        hidden_dim  = data_cfg["hidden_dim"],
        n_heads     = data_cfg["n_heads"],
        drop_p      = data_cfg["drop_p"],
        n_class     = model_cfg["n_class"],
        representation_size = None,
    ).to(device)

    train_config = config["train"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])

    epochs = train_config["epochs"]
    best_val_loss = float('inf')
    precision_metric, recall_metric, f1_metric = get_metrics(n_class=model_cfg["n_class"], device=device)

    train_history = {"loss": [], "acc": []}
    val_history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        # Training ...
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        # Validating ...
        val_loss, val_acc, metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            precision_metric=precision_metric,
            recall_metric=recall_metric,
            f1_metric=f1_metric
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_path, class_names)

        train_history["loss"].append(train_loss)
        train_history["acc"].append(train_acc)

        val_history["loss"].append(val_loss)
        val_history["acc"].append(val_acc)

        print_epoch_result(
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    metrics=metrics
                )
        
if __name__ == "__main__":
    main()