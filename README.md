# DA6401 Assignment 2 — Visual Perception Pipeline

## Student
**Apurv Kshirsagar** | CE22B042 | IIT Madras

## Links
- **WandB Report**: [ADD YOUR PUBLIC WANDB REPORT LINK HERE]
- **GitHub Repo**: https://github.com/ApurvKshirsagar/da6401_assignment_2

## Results (Autograder: 50/50)
| Task | Metric | Score |
|------|--------|-------|
| Classification | Macro-F1 | 0.82 |
| Localization | Acc@IoU=0.5 | 90% |
| Localization | Acc@IoU=0.75 | 50% |
| Segmentation | Macro-Dice | 0.82 |

## Architecture
- **VGG11** from scratch with BatchNorm2d/BatchNorm1d
- **CustomDropout** using inverted dropout (no nn.Dropout)
- **IoULoss** custom implementation in [0,1] range
- **U-Net** decoder with ConvTranspose2d upsampling and skip connections
- **MultiTask** model with 3 dedicated encoders (one per task)

## Design Choices
- BatchNorm placed after every Conv2d and before ReLU for training stability
- CustomDropout applied only in FC layers — conv layers have spatial redundancy
- Segmentation loss = CrossEntropyLoss + DiceLoss to handle class imbalance
- Each task uses its own fine-tuned encoder in the multitask model

## Usage

### Training
```bash
# Task 1: Classification
python train.py --task classify --data_root data/oxford_pets --epochs 40 --batch_size 32 --lr 1e-3

# Task 2: Localization
python train.py --task localize --data_root data/oxford_pets --epochs 40 --batch_size 32 --lr 1e-4

# Task 3: Segmentation
python train.py --task segment --data_root data/oxford_pets --epochs 25 --batch_size 16 --lr 1e-4
```

### Inference
```bash
# Single image
python inference.py --mode single --image path/to/image.jpg --save output.png

# Dataset evaluation
python inference.py --mode evaluate --data_root data/oxford_pets --split test
```

## Project Structure

da6401_assignment_2/
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py
├── losses/
│   └── iou_loss.py
├── models/
│   ├── layers.py
│   ├── vgg11.py
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py
│   └── multitask.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md

## Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```