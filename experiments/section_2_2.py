"""Section 2.2: Dropout ablation — No Dropout vs p=0.2 vs p=0.5"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class ClassifierWithDropout(nn.Module):
    """VGG11 classifier with configurable dropout probability."""

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        # Build encoder manually with custom dropout_p
        self.encoder = VGG11Encoder()
        # Replace the CustomDropout layers with the specified p
        # layers 3 and 7 in encoder.fc are CustomDropout
        self.encoder.fc[3] = CustomDropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.encoder.fc[7] = CustomDropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.head = nn.Linear(4096, 37)

    def forward(self, x):
        return self.head(self.encoder(x))


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


def run_experiment(dropout_p: float, epochs: int = 20):
    name   = {0.0: "no_dropout", 0.2: "dropout_p0.2", 0.5: "dropout_p0.5"}[dropout_p]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project = "da6401-assignment2",
        name    = f"2.2_{name}",
        group   = "section_2_2",
        config  = {"dropout_p": dropout_p, "epochs": epochs, "lr": 1e-3},
    )

    train_ds     = OxfordIIITPetDataset("data/oxford_pets", split="train")
    val_ds       = OxfordIIITPetDataset("data/oxford_pets", split="val")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model     = ClassifierWithDropout(dropout_p=dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_f1   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        gap = train_loss - val_loss   # generalization gap

        wandb.log({
            "epoch":      epoch,
            "train/loss": train_loss,
            "train/f1":   train_f1,
            "val/loss":   val_loss,
            "val/f1":     val_f1,
            "generalization_gap": gap,
        })

        print(f"[{name}] Epoch {epoch:02d} | Train {train_loss:.4f} F1 {train_f1:.4f} | Val {val_loss:.4f} F1 {val_f1:.4f} | Gap {gap:.4f}")

    wandb.finish()


if __name__ == "__main__":
    for p in [0.0, 0.2, 0.5]:
        print(f"\n=== Dropout p={p} ===")
        run_experiment(dropout_p=p, epochs=20)
    print("\nDone!")