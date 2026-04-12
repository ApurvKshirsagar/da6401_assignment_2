"""Section 2.5: Bounding box prediction table with IoU and confidence."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import wandb
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskPerceptionModel(
    classifier_path="checkpoints/classifier.pth",
    localizer_path="checkpoints/localizer.pth",
    unet_path="checkpoints/unet.pth",
).to(device)
model.eval()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

BREEDS = [
    "Abyssinian","American Bulldog","American Pit Bull Terrier",
    "Basset Hound","Beagle","Bengal","Birman","Bombay","Boxer",
    "British Shorthair","Chihuahua","Egyptian Mau","English Cocker Spaniel",
    "English Setter","German Shorthaired","Great Pyrenees","Havanese",
    "Japanese Chin","Keeshond","Leonberger","Maine Coon","Miniature Pinscher",
    "Newfoundland","Persian","Pomeranian","Pug","Ragdoll","Russian Blue",
    "Saint Bernard","Samoyed","Scottish Terrier","Shiba Inu","Siamese",
    "Sphynx","Staffordshire Bull Terrier","Wheaten Terrier","Yorkshire Terrier"
]

def compute_iou(pred, gt):
    px1,py1 = pred[0]-pred[2]/2, pred[1]-pred[3]/2
    px2,py2 = pred[0]+pred[2]/2, pred[1]+pred[3]/2
    gx1,gy1 = gt[0]-gt[2]/2,    gt[1]-gt[3]/2
    gx2,gy2 = gt[0]+gt[2]/2,    gt[1]+gt[3]/2
    ix1,iy1 = max(px1,gx1), max(py1,gy1)
    ix2,iy2 = min(px2,gx2), min(py2,gy2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    pa,ga  = pred[2]*pred[3], gt[2]*gt[3]
    return inter/(pa+ga-inter+1e-6)

def denormalize(tensor):
    return (tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0,1)

def draw_boxes(img_tensor, pred_box, gt_box):
    """Draw GT (green) and pred (red) boxes on image. Returns PIL image."""
    img_np = (denormalize(img_tensor).permute(1,2,0).numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    # GT box — green
    cx,cy,w,h = gt_box
    x1,y1,x2,y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    draw.rectangle([x1,y1,x2,y2], outline="green", width=3)
    # Pred box — red
    cx,cy,w,h = pred_box
    x1,y1,x2,y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
    return img_pil

# Collect 15 samples with real bboxes
ds     = OxfordIIITPetDataset("data/oxford_pets", split="val")
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

results = []
for batch in loader:
    if len(results) >= 15:
        break
    has_bbox = batch["has_bbox"][0].item()
    if not has_bbox:
        continue
    gt_box = batch["bbox"][0].numpy()
    if gt_box[2] == 224 and gt_box[3] == 224:
        continue

    images = batch["image"].to(device)
    with torch.no_grad():
        out = model(images)

    cls_logits  = out["classification"][0].cpu()
    cls_probs   = torch.softmax(cls_logits, dim=0)
    breed_idx   = cls_probs.argmax().item()
    confidence  = cls_probs[breed_idx].item()
    pred_box    = out["localization"][0].cpu().numpy()
    iou         = compute_iou(pred_box, gt_box)

    img_with_boxes = draw_boxes(batch["image"][0], pred_box, gt_box)

    results.append({
        "name":       batch["name"][0],
        "image":      img_with_boxes,
        "breed":      BREEDS[breed_idx] if breed_idx < len(BREEDS) else f"Class_{breed_idx}",
        "confidence": round(confidence * 100, 1),
        "iou":        round(float(iou), 3),
        "pred_box":   [round(float(x),1) for x in pred_box],
        "gt_box":     [round(float(x),1) for x in gt_box],
    })

# Sort by IoU to easily find failure cases
results.sort(key=lambda x: x["iou"])

wandb.init(
    project = "da6401-assignment2",
    name    = "2.5_bbox_predictions",
    group   = "section_2_5",
)

# Log table
table = wandb.Table(columns=[
    "Image","Name","Predicted Breed","Confidence (%)","IoU","GT Box [cx,cy,w,h]","Pred Box [cx,cy,w,h]","Assessment"
])

for r in results:
    if r["iou"] >= 0.5:
        assessment = "Good"
    elif r["iou"] >= 0.3:
        assessment = "Partial"
    else:
        assessment = "Failure"

    table.add_data(
        wandb.Image(r["image"], caption=f"{r['name']} | IoU={r['iou']}"),
        r["name"],
        r["breed"],
        r["confidence"],
        r["iou"],
        str(r["gt_box"]),
        str(r["pred_box"]),
        assessment,
    )

wandb.log({"bbox_predictions": table})

# Also log failure case separately
failure = results[0]  # lowest IoU
print(f"\nFailure case: {failure['name']}")
print(f"  Confidence : {failure['confidence']}%")
print(f"  IoU        : {failure['iou']}")
print(f"  GT box     : {failure['gt_box']}")
print(f"  Pred box   : {failure['pred_box']}")

best = results[-1]  # highest IoU
print(f"\nBest case: {best['name']}")
print(f"  Confidence : {best['confidence']}%")
print(f"  IoU        : {best['iou']}")

wandb.finish()
print("\nDone! Check W&B for the bbox prediction table.")