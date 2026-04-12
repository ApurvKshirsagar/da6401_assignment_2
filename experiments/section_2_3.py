"""Section 2.3: Transfer learning strategies for segmentation."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet


def dice_loss(pred, target, num_classes=3, eps=1e-6):
    probs     = torch.softmax(pred, dim=1)
    target_oh = nn.functional.one_hot(target, num_classes).permute(0,3,1,2).float()
    dims      = (0, 2, 3)
    inter     = (probs * target_oh).sum(dims)
    union     = probs.sum(dims) + target_oh.sum(dims)
    dice      = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def compute_metrics(preds, masks):
    """Compute pixel accuracy and per-class dice."""
    pix_acc = (preds == masks).float().mean().item()
    dice_scores = []
    for c in range(3):
        p     = (preds == c).float()
        t     = (masks  == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if union > 0:
            dice_scores.append((2 * inter / (union + 1e-6)).item())
        else:
            dice_scores.append(1.0)
    return pix_acc, sum(dice_scores) / len(dice_scores)


def run_experiment(strategy: str, epochs: int = 20):
    """
    strategy: 'frozen' | 'partial' | 'full'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project = "da6401-assignment2",
        name    = f"2.3_{strategy}",
        group   = "section_2_3",
        config  = {"strategy": strategy, "epochs": epochs, "lr": 1e-4},
    )

    train_ds     = OxfordIIITPetDataset("data/oxford_pets", split="train")
    val_ds       = OxfordIIITPetDataset("data/oxford_pets", split="val")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Load model with pretrained encoder
    model = VGG11UNet(num_classes=3).to(device)
    ckpt  = torch.load("checkpoints/classifier.pth", map_location=device)
    sd    = ckpt.get("state_dict", ckpt)
    enc_sd = {k.replace("encoder.", ""): v
              for k, v in sd.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_sd, strict=False)

    # Apply freezing strategy
    if strategy == "frozen":
        # Freeze entire encoder
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Strategy: FROZEN — entire encoder frozen")

    elif strategy == "partial":
        # Freeze blocks 1-3, unfreeze blocks 4-5 and FC
        for p in model.encoder.parameters():
            p.requires_grad = False
        for block in [model.encoder.block4, model.encoder.block5,
                      model.encoder.pool4, model.encoder.pool5]:
            for p in block.parameters():
                p.requires_grad = True
        print("Strategy: PARTIAL — blocks 1-3 frozen, 4-5 unfrozen")

    elif strategy == "full":
        # All parameters trainable
        for p in model.encoder.parameters():
            p.requires_grad = True
        print("Strategy: FULL — entire network trainable")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    ce_loss   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[{strategy}] Train", leave=False):
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss, pix_accs, dice_scores = 0.0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{strategy}] Val  ", leave=False):
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)
                logits = model(images)
                loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item()
                preds  = logits.argmax(dim=1)
                pa, dc = compute_metrics(preds.cpu(), masks.cpu())
                pix_accs.append(pa)
                dice_scores.append(dc)

        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        avg_dice  = sum(dice_scores) / len(dice_scores)
        avg_pix   = sum(pix_accs)    / len(pix_accs)
        lr        = optimizer.param_groups[0]["lr"]

        print(f"[{strategy}] Epoch {epoch:02d} | Train {avg_train:.4f} | Val {avg_val:.4f} | Dice {avg_dice:.4f} | PixAcc {avg_pix:.4f}")

        wandb.log({
            "epoch":          epoch,
            "train/loss":     avg_train,
            "val/loss":       avg_val,
            "val/dice":       avg_dice,
            "val/pixel_acc":  avg_pix,
            "lr":             lr,
        })

    wandb.finish()


if __name__ == "__main__":
    for strategy in ["frozen", "partial", "full"]:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*50}")
        run_experiment(strategy=strategy, epochs=20)
    print("\nAll 3 strategies done!")