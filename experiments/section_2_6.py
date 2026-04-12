"""Section 2.6: Segmentation evaluation — Dice vs Pixel Accuracy."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

SEG_COLORS = np.array([
    [0,   200, 0  ],   # class 0 = fg    → green
    [0,   0,   200],   # class 1 = bg    → blue
    [200, 200, 0  ],   # class 2 = border→ yellow
], dtype=np.uint8)

def denormalize(t):
    return (t * IMAGENET_STD + IMAGENET_MEAN).clamp(0,1)

def dice_loss(pred, target, num_classes=3, eps=1e-6):
    probs     = torch.softmax(pred, dim=1)
    target_oh = nn.functional.one_hot(target, num_classes).permute(0,3,1,2).float()
    dims      = (0,2,3)
    inter     = (probs * target_oh).sum(dims)
    union     = probs.sum(dims) + target_oh.sum(dims)
    return 1.0 - ((2*inter+eps)/(union+eps)).mean()

def compute_metrics(preds, masks):
    pix_acc = (preds == masks).float().mean().item()
    dice_scores = []
    for c in range(3):
        p = (preds==c).float(); t = (masks==c).float()
        inter = (p*t).sum(); union = p.sum()+t.sum()
        dice_scores.append((2*inter/(union+1e-6)).item() if union>0 else 1.0)
    return pix_acc, sum(dice_scores)/3

def mask_to_rgb(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        rgb[mask_np==c] = SEG_COLORS[c]
    return rgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG11UNet(num_classes=3).to(device)
ckpt  = torch.load("checkpoints/unet.pth", map_location=device)
sd    = ckpt.get("state_dict", ckpt)
model.load_state_dict(sd, strict=True)
model.eval()

val_ds     = OxfordIIITPetDataset("data/oxford_pets", split="val")
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                        num_workers=0, pin_memory=True)

wandb.init(
    project = "da6401-assignment2",
    name    = "2.6_segmentation_eval",
    group   = "section_2_6",
)

# Collect per-epoch-equivalent metrics by evaluating on val
# (We have one trained model — show metrics across batches to simulate progression)
# For the 5-image visualization, collect first 5 samples

sample_images = []
all_pix, all_dice = [], []
ce_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1)

        for i in range(len(images)):
            pa, dc = compute_metrics(preds[i].cpu(), masks[i].cpu())
            all_pix.append(pa)
            all_dice.append(dc)

            if len(sample_images) < 5:
                orig_np  = (denormalize(images[i].cpu()).permute(1,2,0).numpy()*255).astype(np.uint8)
                gt_rgb   = mask_to_rgb(masks[i].cpu().numpy())
                pred_rgb = mask_to_rgb(preds[i].cpu().numpy())
                sample_images.append((orig_np, gt_rgb, pred_rgb, pa, dc))

print(f"\nOverall Val Pixel Accuracy : {np.mean(all_pix):.4f}")
print(f"Overall Val Macro Dice     : {np.mean(all_dice):.4f}")

# Create 5-image visualization figure
fig, axes = plt.subplots(5, 3, figsize=(12, 20))
col_titles = ["Original Image", "Ground Truth Trimap", "Predicted Trimap"]
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=13, fontweight="bold")

for row, (orig, gt, pred, pa, dc) in enumerate(sample_images):
    axes[row][0].imshow(orig)
    axes[row][0].set_ylabel(f"PixAcc={pa:.3f}\nDice={dc:.3f}", fontsize=9)
    axes[row][1].imshow(gt)
    axes[row][2].imshow(pred)
    for col in range(3):
        axes[row][col].axis("off")

# Legend
legend_patches = [
    plt.Rectangle((0,0),1,1, color=np.array([0,200,0])/255,  label="Foreground (0)"),
    plt.Rectangle((0,0),1,1, color=np.array([0,0,200])/255,  label="Background (1)"),
    plt.Rectangle((0,0),1,1, color=np.array([200,200,0])/255,label="Border (2)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=11)
plt.suptitle("Segmentation Results: Original | GT Trimap | Predicted Trimap", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

os.makedirs("experiments/outputs", exist_ok=True)
fig.savefig("experiments/outputs/seg_5samples.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved: experiments/outputs/seg_5samples.png")

# Log to W&B
wandb.log({
    "segmentation_samples":  wandb.Image("experiments/outputs/seg_5samples.png"),
    "val/pixel_accuracy_mean": float(np.mean(all_pix)),
    "val/dice_mean":           float(np.mean(all_dice)),
    "pixel_acc_distribution":  wandb.Histogram(all_pix),
    "dice_distribution":       wandb.Histogram(all_dice),
})

# Log per-sample scatter to show Dice vs PixAcc discrepancy
scatter_table = wandb.Table(columns=["sample_idx","pixel_accuracy","dice_score","difference"])
for i,(pa,dc) in enumerate(zip(all_pix, all_dice)):
    scatter_table.add_data(i, round(pa,4), round(dc,4), round(pa-dc,4))
wandb.log({"dice_vs_pixacc": scatter_table})

wandb.finish()
print("Done! Check W&B for segmentation visualization.")