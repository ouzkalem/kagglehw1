"""
🛡️ Milestone 2: Data Augmentation & Regularization
====================================================

📝 LEARN: The Overfitting Problem
   With only 500 images per class, our CNN can easily MEMORIZE the training
   data instead of learning general patterns. Signs of overfitting:
     - Training accuracy: 95%  →  "I remember every image!"
     - Validation accuracy: 20% → "I can't recognize new images..."

   Two powerful weapons against overfitting:
   1. DATA AUGMENTATION: Show the model artificially modified versions
      of each image, so it learns to be invariant to transformations.
   2. REGULARIZATION: Add penalties or noise to prevent the model from
      becoming too confident in its memorized patterns.

Key Question: Does the gap between Training Loss and Validation Loss shrink?
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_cifar100_dataloaders
from utils.training import train_model, plot_training_history


# ============================================================================
# 📝 THE MODEL: AugmentedCNN (with Dropout)
# ============================================================================
class AugmentedCNN(nn.Module):
    """
    📝 LEARN: Same architecture as Milestone 1, but with DROPOUT
    ─────────────────────────────────────────────────────────────
    Dropout randomly "turns off" neurons during training (sets them to 0).
    This forces the network to:
      - Not rely on any single neuron
      - Learn redundant representations
      - Act like an ensemble of smaller networks

    Dropout(p=0.25) means each neuron has a 25% chance of being "dropped."
    During evaluation, all neurons are used (but outputs are scaled down).
    """

    def __init__(self, num_classes=100):
        super(AugmentedCNN, self).__init__()

        # Same conv blocks as Milestone 1, but with Dropout after each block
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.1)    # 📝 Spatial dropout: drops entire feature maps
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2)    # 📝 Slightly higher dropout in deeper layers
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # 📝 Ensures fixed output size regardless of input
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),        # 📝 Heavy dropout before final layer
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


def main():
    print("=" * 60)
    print("  🛡️  Milestone 2: Data Augmentation & Regularization")
    print("=" * 60)

    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load Data WITH Augmentation ────────────────────
    print("\n📦 Loading CIFAR-100 WITH data augmentation...")
    print("   📝 Augmentation transforms applied:")
    print("      - RandomHorizontalFlip (50% chance)")
    print("      - RandomCrop with padding (shifts the image)")
    print("      - ColorJitter (brightness, contrast, saturation)")
    print("      - RandomRotation (±15 degrees)")
    print("   These make each epoch show slightly different images!")

    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=BATCH_SIZE,
        augment=True,        # ✅ Augmentation ON!
        image_size=32,
    )

    # ── Create Model with Dropout ──────────────────────
    print("\n🏗️  Building AugmentedCNN (with Dropout)...")
    model = AugmentedCNN(num_classes=100).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Train ──────────────────────────────────────────
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        save_path='models/milestone2_best.pth'
    )

    # ── Plot ───────────────────────────────────────────
    plot_training_history(
        history,
        title="Milestone 2: Augmentation + Dropout",
        save_path='results/milestone2_curves.png'
    )

    # ── Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📝 Milestone 2 Summary")
    print("=" * 60)
    print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")
    print()
    print("  🔍 Compare with Milestone 1:")
    print("    - Is the train/val gap SMALLER? → Augmentation is working!")
    print("    - Is val accuracy HIGHER? → The model generalizes better!")
    print("    - Check: Does train accuracy rise slower? That's GOOD!")
    print("      It means augmentation makes training harder, preventing memorization.")
    print()
    print("  ➡️  Next: Milestone 3 — Hyperparameter Tuning!")
    print("=" * 60)


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()
