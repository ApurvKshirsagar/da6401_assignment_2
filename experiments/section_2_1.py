"""Section 2.1: BatchNorm ablation + activation distribution plots."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.vgg11 import VGG11Encoder, VGG11EncoderNoBN
from models.layers import CustomDropout


class ClassifierWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG11Encoder()
        self.head    = nn.Linear(4096, 37)
    def forward(self, x):
        return self.head(self.encoder(x))


class ClassifierNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG11EncoderNoBN()
        self.head    = nn.Linear(4096, 37)
    def forward(self, x):
        return self.head(self.encoder(x))


def get_block3_activations(model, x, use_bn):
    """Extract activations after block3 conv layers."""
    activations = []
    def hook(module, input, output):
        activations.append(output.detach().cpu())

    if use_bn:
        handle = model.encoder.block3[3].register_forward_hook(hook)
    else:
        handle = model.encoder.block3[2].register_forward_hook(hook)

    with torch.no_grad():
        model(x)
    handle.remove()
    return activations[0]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1


def run_experiment(use_bn: bool, epochs: int = 15, lr: float = 1e-3):
    name   = "with_batchnorm" if use_bn else "no_batchnorm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project = "da6401-assignment2",
        name    = f"2.1_{name}",
        group   = "section_2_1",
        config  = {"use_bn": use_bn, "epochs": epochs, "lr": lr},
    )

    train_ds = OxfordIIITPetDataset("data/oxford_pets", split="train")
    val_ds   = OxfordIIITPetDataset("data/oxford_pets", split="val")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model     = (ClassifierWithBN() if use_bn else ClassifierNoBN()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Get one fixed batch for activation visualization
    fixed_batch = next(iter(val_loader))
    fixed_img   = fixed_batch["image"][:8].to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_f1   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Log activation distribution at epochs 1, 5, 10, 15
        if epoch in (1, 5, 10, 15):
            acts = get_block3_activations(model, fixed_img, use_bn)
            acts_flat = acts.numpy().flatten()
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/f1": train_f1,
                "val/loss":   val_loss,   "val/f1":   val_f1,
                f"activations/block3_epoch{epoch}": wandb.Histogram(acts_flat),
                "activations/block3_mean": float(acts_flat.mean()),
                "activations/block3_std":  float(acts_flat.std()),
            })
        else:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/f1": train_f1,
                "val/loss":   val_loss,   "val/f1":   val_f1,
            })

        print(f"[{name}] Epoch {epoch:02d} | Train loss {train_loss:.4f} F1 {train_f1:.4f} | Val loss {val_loss:.4f} F1 {val_f1:.4f}")

    wandb.finish()


if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)
    print("=== Run 1: WITH BatchNorm ===")
    run_experiment(use_bn=True,  epochs=15, lr=1e-3)
    print("=== Run 2: WITHOUT BatchNorm ===")
    run_experiment(use_bn=False, epochs=15, lr=1e-3)
    print("Done! Check W&B for plots.")