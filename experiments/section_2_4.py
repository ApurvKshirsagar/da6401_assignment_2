"""Section 2.4: Feature map visualization from first and last conv layers."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import wandb
from PIL import Image
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained classifier
model = VGG11Classifier(num_classes=37).to(device)
ckpt  = torch.load("checkpoints/classifier.pth", map_location=device)
sd    = ckpt.get("state_dict", ckpt)
model.load_state_dict(sd, strict=True)
model.eval()

# Get one dog image from val set
ds = OxfordIIITPetDataset("data/oxford_pets", split="val")
dog_sample = None
for i in range(len(ds)):
    s = ds[i]
    # species 2 = dog (class_id >= 12 roughly, but easier to just grab first available)
    if s["name"].split("_")[0] in [
        "american_bulldog", "american_pit_bull_terrier", "basset_hound",
        "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees",
        "havanese", "japanese_chin", "keeshond", "leonberger",
        "miniature_pinscher", "newfoundland", "pomeranian", "pug",
        "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
        "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"
    ]:
        dog_sample = s
        break

if dog_sample is None:
    dog_sample = ds[0]

img_tensor = dog_sample["image"].unsqueeze(0).to(device)  # [1, 3, 224, 224]
print(f"Using image: {dog_sample['name']}")

# Hook to capture feature maps
first_conv_maps = []
last_conv_maps  = []

def hook_first(module, input, output):
    first_conv_maps.append(output.detach().cpu())

def hook_last(module, input, output):
    last_conv_maps.append(output.detach().cpu())

# First conv = encoder.block1[0] (Conv2d 3→64)
# Last conv before pool = encoder.block5[3] (Conv2d 512→512)
h1 = model.encoder.block1[0].register_forward_hook(hook_first)
h2 = model.encoder.block5[3].register_forward_hook(hook_last)

with torch.no_grad():
    _ = model(img_tensor)

h1.remove()
h2.remove()

first_maps = first_conv_maps[0][0]   # [64, 224, 224]
last_maps  = last_conv_maps[0][0]    # [512, 14, 14]

print(f"First conv feature maps shape: {first_maps.shape}")
print(f"Last  conv feature maps shape: {last_maps.shape}")

# Visualize first 16 feature maps from each layer
def plot_feature_maps(maps, title, n=16, save_path=None):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < n and i < maps.shape[0]:
            fm = maps[i].numpy()
            # Normalize to [0,1] for visualization
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            ax.imshow(fm, cmap="viridis")
            ax.set_title(f"Filter {i+1}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

os.makedirs("experiments/outputs", exist_ok=True)

plot_feature_maps(
    first_maps, 
    f"First Conv Layer Feature Maps (64 filters, 224×224)\nImage: {dog_sample['name']}",
    save_path="experiments/outputs/first_conv_maps.png"
)

plot_feature_maps(
    last_maps,
    f"Last Conv Layer Feature Maps (512 filters, 14×14)\nImage: {dog_sample['name']}",
    save_path="experiments/outputs/last_conv_maps.png"
)

# Also save the original image
orig_img = Image.open(f"data/oxford_pets/images/{dog_sample['name']}.jpg").convert("RGB")
orig_img_resized = orig_img.resize((224, 224))
orig_img_resized.save("experiments/outputs/original_image.png")

# Log to W&B
wandb.init(
    project = "da6401-assignment2",
    name    = "2.4_feature_maps",
    group   = "section_2_4",
)

wandb.log({
    "original_image":        wandb.Image("experiments/outputs/original_image.png",
                                          caption=f"Input: {dog_sample['name']}"),
    "first_conv_feature_maps": wandb.Image("experiments/outputs/first_conv_maps.png",
                                            caption="First Conv Layer — 16 of 64 filters (224×224)"),
    "last_conv_feature_maps":  wandb.Image("experiments/outputs/last_conv_maps.png",
                                            caption="Last Conv Layer — 16 of 512 filters (14×14)"),
})

wandb.finish()
print("\nDone! Check W&B for feature map images and experiments/outputs/ for local files.")