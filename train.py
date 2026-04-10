"""Training entrypoint
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Training")
    parser.add_argument("--task",        type=str,   default="classify",
                        choices=["classify", "localize", "segment"],
                        help="Which task to train")
    parser.add_argument("--data_root",   type=str,   default="data/oxford_pets")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights (for localize task)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, task):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        optimizer.zero_grad()

        if task == "classify":
            labels = batch["label"].to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        elif task == "localize":
            bboxes     = batch["bbox"].to(device)
            has_bbox   = batch["has_bbox"].to(device)
            pred_boxes = model(images)

            # Only compute loss on samples that have real bboxes
            if has_bbox.any():
                p = pred_boxes[has_bbox]
                t = bboxes[has_bbox]
                mse_loss = criterion["mse"](p / 224.0, t / 224.0)
                iou_loss = criterion["iou"](p, t)
                loss     = mse_loss + iou_loss
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if task == "classify":
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, f1
    return avg_loss, None


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    iou_scores = []

    for batch in tqdm(loader, desc="Val  ", leave=False):
        images = batch["image"].to(device)

        if task == "classify":
            labels = batch["label"].to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        elif task == "localize":
            bboxes     = batch["bbox"].to(device)
            has_bbox   = batch["has_bbox"].to(device)
            pred_boxes = model(images)

            if has_bbox.any():
                p = pred_boxes[has_bbox]
                t = bboxes[has_bbox]
                mse_loss = criterion["mse"](p / 224.0, t / 224.0)
                iou_loss = criterion["iou"](p, t)
                loss     = mse_loss + iou_loss
                # Track IoU score (1 - loss) for logging
                iou_scores.extend((1 - criterion["iou"](p, t).item()
                                    if False else
                                    IoULoss(reduction="none")(p, t)
                                    .cpu().numpy().tolist()))
            else:
                loss = torch.tensor(0.0)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if task == "classify":
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return avg_loss, f1, acc

    mean_iou = float(np.mean([1 - s for s in iou_scores])) if iou_scores else 0.0
    return avg_loss, mean_iou, 0.0


# ─────────────────────────────────────────────────────────────────────────────
def train_classifier(args, device):
    """Train Task 1: VGG11 classification."""

    # Datasets
    train_ds = OxfordIIITPetDataset(args.data_root, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model     = VGG11Classifier(num_classes=37).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_f1   = 0.0
    save_path = os.path.join(args.checkpoint_dir, "classifier.pth")

    print(f"\nTraining classifier for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, "classify"
        )
        val_loss, val_f1, val_acc = evaluate(
            model, val_loader, criterion, device, "classify"
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f}  F1: {val_f1:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )

        wandb.log({
            "epoch":           epoch,
            "train/loss":      train_loss,
            "train/f1":        train_f1,
            "val/loss":        val_loss,
            "val/f1":          val_f1,
            "val/accuracy":    val_acc,
            "lr":              lr,
        })

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "state_dict":   model.state_dict(),
                "epoch":        epoch,
                "best_metric":  best_f1,
                "arch":         "VGG11Classifier",
            }, save_path)
            print(f"  → Saved best classifier (F1={best_f1:.4f})")

    print(f"\nBest val F1: {best_f1:.4f} — checkpoint at {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
def train_localizer(args, device):
    """Train Task 2: VGG11 localization."""

    train_ds = OxfordIIITPetDataset(args.data_root, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Load encoder weights from classifier if available
    classifier_path = os.path.join(args.checkpoint_dir, "classifier.pth")
    model = VGG11Localizer().to(device)

    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=device)
        sd   = ckpt.get("state_dict", ckpt)
        # Load only encoder weights
        encoder_sd = {
            k.replace("encoder.", ""): v
            for k, v in sd.items() if k.startswith("encoder.")
        }
        missing = model.encoder.load_state_dict(encoder_sd, strict=False)
        print(f"Loaded encoder from classifier.pth — {missing}")

        if args.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("Encoder frozen.")
        else:
            print("Encoder will be fine-tuned.")
    else:
        print("No classifier.pth found — training localizer from scratch.")

    criterion = {
        "mse": nn.MSELoss(),
        "iou": IoULoss(reduction="mean"),
    }

    # Only optimize unfrozen params
    params    = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_iou  = 0.0
    save_path = os.path.join(args.checkpoint_dir, "localizer.pth")

    print(f"\nTraining localizer for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device, "localize"
        )
        val_loss, val_iou, _ = evaluate(
            model, val_loader, criterion, device, "localize"
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}  IoU: {val_iou:.4f} | "
            f"LR: {lr:.6f}"
        )

        wandb.log({
            "epoch":        epoch,
            "train/loss":   train_loss,
            "val/loss":     val_loss,
            "val/iou":      val_iou,
            "lr":           lr,
        })

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "state_dict":  model.state_dict(),
                "epoch":       epoch,
                "best_metric": best_iou,
                "arch":        "VGG11Localizer",
            }, save_path)
            print(f"  → Saved best localizer (IoU={best_iou:.4f})")

    print(f"\nBest val IoU: {best_iou:.4f} — checkpoint at {save_path}")
    return model

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              num_classes: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss averaged over classes.

    Args:
        pred:   [B, C, H, W] logits
        target: [B, H, W] class indices
    """
    probs  = torch.softmax(pred, dim=1)          # [B, C, H, W]
    target_oh = nn.functional.one_hot(
        target, num_classes
    ).permute(0, 3, 1, 2).float()                # [B, C, H, W]

    dims   = (0, 2, 3)   # sum over batch, H, W
    inter  = (probs * target_oh).sum(dims)
    union  = probs.sum(dims) + target_oh.sum(dims)
    dice   = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def train_segmentation(args, device):
    """Train Task 3: U-Net segmentation."""
    from models.segmentation import VGG11UNet

    train_ds = OxfordIIITPetDataset(args.data_root, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = VGG11UNet(num_classes=3).to(device)

    # Load encoder weights from classifier if available
    classifier_path = os.path.join(args.checkpoint_dir, "classifier.pth")
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=device)
        sd   = ckpt.get("state_dict", ckpt)
        encoder_sd = {
            k.replace("encoder.", ""): v
            for k, v in sd.items() if k.startswith("encoder.")
        }
        missing = model.encoder.load_state_dict(encoder_sd, strict=False)
        print(f"Loaded encoder from classifier.pth — {missing}")

        if args.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("Encoder frozen.")
        else:
            print("Encoder will be fine-tuned.")

    ce_loss  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    best_dice = 0.0
    save_path = os.path.join(args.checkpoint_dir, "unet.pth")

    print(f"\nTraining U-Net for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Train", leave=False):
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)       # [B, 224, 224] longs

            optimizer.zero_grad()
            logits = model(images)                  # [B, 3, 224, 224]
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        dice_scores = []
        pixel_correct = 0
        pixel_total   = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val  ", leave=False):
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)

                logits = model(images)
                loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)         # [B, 224, 224]

                # Pixel accuracy
                pixel_correct += (preds == masks).sum().item()
                pixel_total   += masks.numel()

                # Dice per batch
                for c in range(3):
                    p = (preds == c).float()
                    t = (masks  == c).float()
                    inter = (p * t).sum()
                    union = p.sum() + t.sum()
                    if union > 0:
                        dice_scores.append(
                            (2 * inter / (union + 1e-6)).item()
                        )

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        mean_dice = float(sum(dice_scores) / len(dice_scores)) if dice_scores else 0.0
        pix_acc   = pixel_correct / pixel_total

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {avg_train:.4f} | "
            f"Val loss: {avg_val:.4f}  "
            f"Dice: {mean_dice:.4f}  "
            f"PixAcc: {pix_acc:.4f} | "
            f"LR: {lr:.6f}"
        )

        wandb.log({
            "epoch":          epoch,
            "train/loss":     avg_train,
            "val/loss":       avg_val,
            "val/dice":       mean_dice,
            "val/pixel_acc":  pix_acc,
            "lr":             lr,
        })

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                "state_dict":  model.state_dict(),
                "epoch":       epoch,
                "best_metric": best_dice,
                "arch":        "VGG11UNet",
            }, save_path)
            print(f"  → Saved best U-Net (Dice={best_dice:.4f})")

    print(f"\nBest val Dice: {best_dice:.4f} — checkpoint at {save_path}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Init W&B
    wandb.init(
        project = args.wandb_project,
        name    = f"task1_{args.task}_lr{args.lr}_bs{args.batch_size}",
        config  = vars(args),
    )

    if args.task == "classify":
        train_classifier(args, device)
    elif args.task == "localize":
        train_localizer(args, device)
    elif args.task == "segment":
        train_segmentation(args, device)

    wandb.finish()


if __name__ == "__main__":
    main()