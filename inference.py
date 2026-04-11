"""Inference and evaluation
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224

BREEDS = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay",
    "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired",
    "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
    "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue",
    "Saint Bernard", "Samoyed", "Scottish Terish", "Shiba Inu",
    "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier"
]

SEG_COLORS = {
    0: (0.2, 0.8, 0.2, 0.5),   # fg  — green
    1: (0.2, 0.2, 0.8, 0.3),   # bg  — blue
    2: (0.8, 0.8, 0.2, 0.5),   # border — yellow
}


def load_image(image_path: str) -> tuple:
    """Load and preprocess a single image.

    Returns:
        tensor: [1, 3, 224, 224] normalized tensor
        original: PIL Image (for visualization)
    """
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    original = Image.open(image_path).convert("RGB")
    img_np   = np.array(original)
    out      = transform(image=img_np)
    tensor   = out["image"].float().unsqueeze(0)   # [1, 3, 224, 224]
    return tensor, original


def run_inference(
    image_path:      str,
    classifier_path: str = "checkpoints/classifier.pth",
    localizer_path:  str = "checkpoints/localizer.pth",
    unet_path:       str = "checkpoints/unet.pth",
    device:          str = "auto",
    visualize:       bool = True,
    save_path:       str  = None,
) -> dict:
    """Run full pipeline inference on a single image.

    Args:
        image_path:      Path to input image.
        classifier_path: Path to classifier checkpoint.
        localizer_path:  Path to localizer checkpoint.
        unet_path:       Path to unet checkpoint.
        device:          'auto', 'cuda', or 'cpu'.
        visualize:       Whether to show/save visualization.
        save_path:       If set, save visualization to this path.

    Returns:
        dict with keys:
            'breed':      predicted breed name (str)
            'breed_idx':  predicted class index (int)
            'confidence': softmax confidence (float)
            'bbox':       [cx, cy, w, h] in pixel space (list)
            'mask':       [224, 224] numpy array (0=fg, 1=bg, 2=border)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model
    model = MultiTaskPerceptionModel(
        classifier_path=classifier_path,
        localizer_path=localizer_path,
        unet_path=unet_path,
    ).to(device)
    model.eval()

    # Load image
    tensor, original = load_image(image_path)
    tensor = tensor.to(device)

    # Run inference
    with torch.no_grad():
        out = model(tensor)

    # Classification
    cls_logits  = out["classification"][0]           # [37]
    cls_probs   = torch.softmax(cls_logits, dim=0)
    breed_idx   = cls_probs.argmax().item()
    confidence  = cls_probs[breed_idx].item()
    breed_name  = BREEDS[breed_idx] if breed_idx < len(BREEDS) else f"Class_{breed_idx}"

    # Localization
    bbox = out["localization"][0].cpu().numpy()      # [cx, cy, w, h]

    # Segmentation
    seg_logits = out["segmentation"][0]              # [3, 224, 224]
    seg_mask   = seg_logits.argmax(dim=0).cpu().numpy()  # [224, 224]

    results = {
        "breed":      breed_name,
        "breed_idx":  breed_idx,
        "confidence": confidence,
        "bbox":       bbox.tolist(),
        "mask":       seg_mask,
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"Image      : {os.path.basename(image_path)}")
    print(f"Breed      : {breed_name} (class {breed_idx})")
    print(f"Confidence : {confidence*100:.1f}%")
    cx, cy, w, h = bbox
    print(f"BBox       : cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f}")
    print(f"Seg classes: {np.unique(seg_mask).tolist()}")
    print(f"{'='*50}\n")

    if visualize:
        _visualize(original, bbox, seg_mask, breed_name, confidence, save_path)

    return results


def _visualize(
    original:   Image.Image,
    bbox:       np.ndarray,
    seg_mask:   np.ndarray,
    breed_name: str,
    confidence: float,
    save_path:  str = None,
):
    """Create and show/save visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original + bounding box
    axes[0].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
    cx, cy, w, h = bbox
    x1, y1 = cx - w / 2, cy - h / 2
    rect = patches.Rectangle(
        (x1, y1), w, h,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"Detection\n{breed_name} ({confidence*100:.1f}%)", fontsize=10)
    axes[0].axis("off")

    # Panel 2: Segmentation mask
    colored = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4))
    for cls_id, color in SEG_COLORS.items():
        colored[seg_mask == cls_id] = color
    axes[1].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[1].imshow(colored)
    axes[1].set_title("Segmentation\n(green=fg, blue=bg, yellow=border)", fontsize=10)
    axes[1].axis("off")

    # Panel 3: Overlay
    axes[2].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[2].imshow(colored)
    rect2 = patches.Rectangle(
        (x1, y1), w, h,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[2].add_patch(rect2)
    axes[2].set_title("Full Pipeline Output", fontsize=10)
    axes[2].axis("off")

    plt.suptitle(f"DA6401 Visual Perception Pipeline — {breed_name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def evaluate_dataset(
    data_root:       str  = "data/oxford_pets",
    split:           str  = "test",
    classifier_path: str  = "checkpoints/classifier.pth",
    localizer_path:  str  = "checkpoints/localizer.pth",
    unet_path:       str  = "checkpoints/unet.pth",
    num_samples:     int  = 100,
):
    """Evaluate pipeline on dataset split and print metrics."""
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    from data.pets_dataset import OxfordIIITPetDataset
    from losses.iou_loss import IoULoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel(
        classifier_path=classifier_path,
        localizer_path=localizer_path,
        unet_path=unet_path,
    ).to(device)
    model.eval()

    ds     = OxfordIIITPetDataset(data_root, split=split)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    iou_scores, dice_scores = [], []
    iou_fn = IoULoss(reduction="none")
    count  = 0

    with torch.no_grad():
        for batch in loader:
            if count >= num_samples:
                break

            images = batch["image"].to(device)
            out    = model(images)

            # Classification
            preds  = out["classification"].argmax(dim=1).cpu().numpy()
            labels = batch["label"].numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Localization
            has_bbox = batch["has_bbox"]
            if has_bbox.any():
                pb = out["localization"].cpu()[has_bbox]
                tb = batch["bbox"][has_bbox]
                iou_loss = iou_fn(pb, tb).numpy()
                iou_scores.extend((1 - iou_loss).tolist())

            # Segmentation
            seg_preds = out["segmentation"].argmax(dim=1).cpu()
            masks     = batch["mask"]
            for c in range(3):
                p = (seg_preds == c).float()
                t = (masks     == c).float()
                inter = (p * t).sum(dim=[1, 2])
                union = p.sum(dim=[1, 2]) + t.sum(dim=[1, 2])
                d = (2 * inter / (union + 1e-6))
                dice_scores.extend(d.numpy().tolist())

            count += len(images)

    f1        = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mean_iou  = float(np.mean(iou_scores)) if iou_scores else 0.0
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

    print(f"\nEvaluation on {split} split ({count} samples):")
    print(f"  Classification Macro-F1 : {f1:.4f}")
    print(f"  Localization  Mean IoU  : {mean_iou:.4f}")
    print(f"  Segmentation  Mean Dice : {mean_dice:.4f}")
    return {"f1": f1, "iou": mean_iou, "dice": mean_dice}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Inference")
    parser.add_argument("--mode",       type=str, default="single",
                        choices=["single", "evaluate"],
                        help="single image inference or dataset evaluation")
    parser.add_argument("--image",      type=str, default=None,
                        help="Path to input image (for single mode)")
    parser.add_argument("--save",       type=str, default=None,
                        help="Path to save visualization")
    parser.add_argument("--data_root",  type=str, default="data/oxford_pets")
    parser.add_argument("--split",      type=str, default="test")
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--classifier_path", type=str,
                        default="checkpoints/classifier.pth")
    parser.add_argument("--localizer_path",  type=str,
                        default="checkpoints/localizer.pth")
    parser.add_argument("--unet_path",       type=str,
                        default="checkpoints/unet.pth")
    args = parser.parse_args()

    if args.mode == "single":
        if args.image is None:
            print("Please provide --image path for single mode")
        else:
            run_inference(
                image_path      = args.image,
                classifier_path = args.classifier_path,
                localizer_path  = args.localizer_path,
                unet_path       = args.unet_path,
                save_path       = args.save,
            )
    elif args.mode == "evaluate":
        evaluate_dataset(
            data_root       = args.data_root,
            split           = args.split,
            classifier_path = args.classifier_path,
            localizer_path  = args.localizer_path,
            unet_path       = args.unet_path,
            num_samples     = args.n_samples,
        )