"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet mean/std for normalization (as required by assignment)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224  # fixed per VGG11 paper


def get_transforms(split: str) -> A.Compose:
    """Return albumentations transform pipeline for a given split.

    Args:
        split: 'train' or 'val' or 'test'

    Returns:
        Albumentations Compose with bbox and mask support.
    """
    if split == "train":
        transforms = [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="coco",          # coco = [x_min, y_min, w, h]
            label_fields=["bbox_labels"],
            min_visibility=0.3,
        ),
    )


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Returns per sample:
        image  : [3, 224, 224] normalized float tensor
        label  : int  (0-indexed breed, 0..36)
        bbox   : [4]  float tensor (cx, cy, w, h) in pixel space
        mask   : [224, 224] long tensor  (1=fg, 2=bg, 3=border → remapped to 0,1,2)
        has_bbox: bool — whether a ground-truth bbox exists for this sample
    """

    def __init__(
        self,
        root:      str,
        split:     str = "train",
        transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            root:      Path to oxford_pets folder
                       (contains images/ and annotations/).
            split:     'train' or 'val' or 'test'
            transform: Optional albumentations Compose. If None,
                       uses default get_transforms(split).
        """
        super().__init__()

        self.root      = root
        self.split     = split
        self.transform = transform if transform is not None else get_transforms(split)

        self.img_dir   = os.path.join(root, "images")
        self.mask_dir  = os.path.join(root, "annotations", "trimaps")
        self.xml_dir   = os.path.join(root, "annotations", "xmls")

        # Load split file
        self.samples = self._load_split()

    # ──────────────────────────────────────────────────────────────────────
    def _load_split(self) -> List[Dict]:
        list_file = os.path.join(self.root, "annotations", "list.txt")
        all_samples = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts    = line.split()
                name     = parts[0]
                class_id = int(parts[1]) - 1
                species  = int(parts[2])
                breed_id = int(parts[3])

                img_path = os.path.join(self.img_dir, name + ".jpg")
                if not os.path.exists(img_path):
                    continue

                all_samples.append({
                    "name":     name,
                    "class_id": class_id,
                    "species":  species,
                    "breed_id": breed_id,
                })

        rng     = np.random.RandomState(42)
        indices = rng.permutation(len(all_samples))
        n       = len(all_samples)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        if self.split == "train":
            chosen = indices[:n_train]
        elif self.split == "val":
            chosen = indices[n_train: n_train + n_val]
        else:
            chosen = indices[n_train + n_val:]

        return [all_samples[i] for i in chosen]

    # ──────────────────────────────────────────────────────────────────────
    def _load_bbox(self, name: str, img_w: int, img_h: int) -> Optional[np.ndarray]:
        """Load bounding box from XML and convert to cxcywh pixel coords.

        Returns [cx, cy, w, h] numpy array, or None if XML missing.
        """
        xml_path = os.path.join(self.xml_dir, name + ".xml")
        if not os.path.exists(xml_path):
            return None

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bndbox = root.find(".//bndbox")
            if bndbox is None:
                return None

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Clamp to image bounds
            xmin = max(0.0, min(xmin, img_w))
            ymin = max(0.0, min(ymin, img_h))
            xmax = max(0.0, min(xmax, img_w))
            ymax = max(0.0, min(ymax, img_h))

            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            w  = xmax - xmin
            h  = ymax - ymin

            return np.array([cx, cy, w, h], dtype=np.float32)

        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    # ──────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> Dict:
        sample   = self.samples[idx]
        name     = sample["name"]
        class_id = sample["class_id"]

        # ── Load image ────────────────────────────────────────────────────
        img = np.array(
            Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        )
        img_h, img_w = img.shape[:2]

        # ── Load mask ─────────────────────────────────────────────────────
        mask_path = os.path.join(self.mask_dir, name + ".png")
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))   # values 1,2,3
        else:
            mask = np.ones((img_h, img_w), dtype=np.uint8)

        # Remap trimap: 1→0 (fg), 2→1 (bg), 3→2 (border)
        mask = (mask - 1).astype(np.uint8)   # now 0,1,2

        # ── Load bbox ─────────────────────────────────────────────────────
        bbox_arr = self._load_bbox(name, img_w, img_h)
        has_bbox = bbox_arr is not None

        if not has_bbox:
            # Dummy bbox covering whole image
            bbox_arr = np.array(
                [img_w / 2, img_h / 2, float(img_w), float(img_h)],
                dtype=np.float32
            )

        # Convert cxcywh → xywh (coco format) for albumentations
        bx = bbox_arr[0] - bbox_arr[2] / 2   # x_min
        by = bbox_arr[1] - bbox_arr[3] / 2   # y_min
        bw = bbox_arr[2]
        bh = bbox_arr[3]

        # Clamp to valid range
        bx = float(np.clip(bx, 0, img_w - 1))
        by = float(np.clip(by, 0, img_h - 1))
        bw = float(np.clip(bw, 1, img_w - bx))
        bh = float(np.clip(bh, 1, img_h - by))

        # ── Apply transforms ──────────────────────────────────────────────
        transformed = self.transform(
            image=img,
            mask=mask,
            bboxes=[[bx, by, bw, bh]],
            bbox_labels=[0],
        )

        image_t = transformed["image"].float()          # [3, 224, 224]
        mask_t = transformed["mask"]
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.from_numpy(mask_t.copy())
        mask_t = mask_t.long()                          # [224, 224]

        # Convert bbox back to cxcywh in the resized (224x224) pixel space
        if len(transformed["bboxes"]) > 0:
            tb = transformed["bboxes"][0]   # [x_min, y_min, w, h] in 224 space
            cx_t = tb[0] + tb[2] / 2
            cy_t = tb[1] + tb[3] / 2
            bbox_t = torch.tensor(
                [cx_t, cy_t, tb[2], tb[3]], dtype=torch.float32
            )
        else:
            # Bbox got cropped out — use full image
            bbox_t = torch.tensor(
                [112.0, 112.0, 224.0, 224.0], dtype=torch.float32
            )
            has_bbox = False

        return {
            "image":    image_t,                              # [3, 224, 224]
            "label":    torch.tensor(class_id, dtype=torch.long),
            "bbox":     bbox_t,                               # [4] cxcywh px
            "mask":     mask_t,                               # [224, 224]
            "has_bbox": torch.tensor(has_bbox, dtype=torch.bool),
            "name":     name,
        }