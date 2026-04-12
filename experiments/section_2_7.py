"""Section 2.7: In-the-wild pipeline showcase."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from inference import run_inference

wandb.init(
    project = "da6401-assignment2",
    name    = "2.7_wild_images",
    group   = "section_2_7",
)

images = [
    "experiments/test_images/wild_pet_1.jpg",
    "experiments/test_images/wild_pet_2.jpg",
    "experiments/test_images/wild_pet_3.jpg",
]

table = wandb.Table(columns=["Image","Predicted Breed","Confidence (%)","BBox [cx,cy,w,h]","Analysis"])

for i, img_path in enumerate(images):
    if not os.path.exists(img_path):
        print(f"Missing: {img_path} — skipping")
        continue

    save_path = img_path.replace(".jpg", "_output.png").replace(".jpeg","_output.png").replace(".png","_output.png")
    results = run_inference(
        image_path = img_path,
        save_path  = save_path,
        visualize  = True,
    )

    table.add_data(
        wandb.Image(save_path, caption=os.path.basename(img_path)),
        results["breed"],
        round(results["confidence"]*100, 1),
        [round(x,1) for x in results["bbox"]],
        "In-the-wild test image",
    )

wandb.log({"wild_image_results": table})
wandb.finish()
print("Done!")