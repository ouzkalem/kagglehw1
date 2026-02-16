"""
🎯 Milestone 1: The Baseline CNN — Building from Scratch
=========================================================

📝 LEARN: What is a CNN (Convolutional Neural Network)?
   A CNN is specifically designed for image processing. Unlike regular neural
   networks that see images as flat lists of numbers, CNNs preserve the
   spatial structure of images using special operations:

   1. CONVOLUTION (Conv2d): A small filter slides across the image,
      detecting patterns like edges, corners, and textures.
      Think of it like a magnifying glass looking for specific features.

   2. POOLING (MaxPool2d): Shrinks the image by keeping only the
      strongest activations. This makes the model invariant to small
      translations and reduces computation.

   3. ACTIVATION (ReLU): Introduces non-linearity.
      Without it, stacking layers would be like having one big linear layer.
      ReLU(x) = max(0, x) — simple but powerful!

   4. FULLY CONNECTED (Linear): Takes the extracted features and
      produces class scores (one per class → 100 for CIFAR-100).

Architecture:
   Input (3x32x32) →
   [Conv→ReLU→Pool] → [Conv→ReLU→Pool] → [Conv→ReLU→Pool] →
   Flatten → [Linear→ReLU] → [Linear] → 100 class scores

Goal: Beat random guessing (1% accuracy = 1/100 classes)
"""

import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_cifar100_dataloaders, CIFAR100_CLASSES
from utils.training import train_model, plot_training_history


# ============================================================================
# 📝 THE MODEL: SimpleCNN
# ============================================================================
class SimpleCNN(nn.Module):
    """
    📝 LEARN: nn.Module — The Base Class for All PyTorch Models
    ───────────────────────────────────────────────────────────
    Every model in PyTorch inherits from nn.Module.
    You need to define:
      1. __init__: Create the layers
      2. forward: Define how data flows through the layers

    This simple CNN has 3 convolutional blocks followed by a classifier.
    """

    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()

        # 📝 LEARN: Convolutional Blocks
        # ──────────────────────────────
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        #   - in_channels:  Number of input channels (3 for RGB)
        #   - out_channels: Number of filters (each learns a different pattern)
        #   - kernel_size:  Size of the sliding filter (3x3 is most common)
        #   - padding=1:    Adds 1 pixel border so output size = input size
        #
        # Why 3x3 filters? Research (VGGNet paper) showed that two 3x3 filters
        # have the same receptive field as one 5x5 filter, but with fewer
        # parameters and more non-linearity (ReLU between them).
        #
        # 📝 RECEPTIVE FIELD: The area of the original image that influences
        # a single neuron in a deeper layer.
        #   - After 1 conv layer (3x3): receptive field = 3x3
        #   - After 2 conv layers (3x3): receptive field = 5x5
        #   - After pooling: receptive field doubles!
        #
        # More layers → larger receptive field → model "sees" bigger patterns

        # Block 1: 3 → 32 channels, then shrink with MaxPool
        # Input:  3 x 32 x 32
        # Output: 32 x 16 x 16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3→32 channels
            nn.BatchNorm2d(32),                            # Normalize activations
            nn.ReLU(inplace=True),                         # Non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2)          # 32x32 → 16x16
        )

        # Block 2: 32 → 64 channels
        # Input:  32 x 16 x 16
        # Output: 64 x 8 x 8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # 16x16 → 8x8
        )

        # Block 3: 64 → 128 channels
        # Input:  64 x 8 x 8
        # Output: 128 x 4 x 4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # 8x8 → 4x4
        )

        # 📝 LEARN: Classifier Head
        # After convolution, we have a 128x4x4 = 2048-dimensional feature vector.
        # We "flatten" this 3D tensor into a 1D vector and pass it through
        # fully connected (Linear) layers to get 100 class scores.
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # 128x4x4 → 2048
            nn.Linear(128 * 4 * 4, 256),     # 2048 → 256
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)       # 256 → 100 (one score per class)
        )

    def forward(self, x):
        """
        📝 LEARN: The Forward Pass
        ─────────────────────────
        Data flows through the network in order.
        x starts as a batch of images: shape [batch_size, 3, 32, 32]
        Each block extracts higher-level features:
          Block 1: Learns edges, basic shapes
          Block 2: Learns textures, patterns
          Block 3: Learns object parts
          Classifier: Maps features to class scores
        """
        x = self.block1(x)       # [B, 32, 16, 16]
        x = self.block2(x)       # [B, 64, 8, 8]
        x = self.block3(x)       # [B, 128, 4, 4]
        x = self.classifier(x)   # [B, 100]
        return x


# ============================================================================
# 📝 MAIN: Putting It All Together
# ============================================================================
def main():
    print("=" * 60)
    print("  🎯 Milestone 1: Baseline CNN from Scratch")
    print("=" * 60)

    # ── Configuration ──────────────────────────────────
    BATCH_SIZE = 64
    NUM_EPOCHS = 15          # Keep low for CPU training
    LEARNING_RATE = 1e-3     # 0.001 — a good starting point
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Step 1: Load Data ──────────────────────────────
    print("\n📦 Step 1: Loading CIFAR-100 dataset...")
    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=BATCH_SIZE,
        augment=False,       # No augmentation yet! (That's Milestone 2)
        image_size=32,       # Native CIFAR-100 size
    )

    # ── Step 2: Create Model ───────────────────────────
    print("\n🏗️  Step 2: Building SimpleCNN...")
    model = SimpleCNN(num_classes=100).to(DEVICE)

    # 📝 LEARN: Let's see what our model looks like!
    print(f"\n📐 Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   That's {total_params * 4 / 1024 / 1024:.1f} MB of weights (float32)")

    # ── Step 3: Define Loss and Optimizer ──────────────
    # 📝 LEARN: CrossEntropyLoss
    # For classification, we use CrossEntropyLoss which:
    #   1. Applies Softmax to convert raw scores → probabilities
    #   2. Computes the negative log-likelihood of the correct class
    # Lower loss = higher probability assigned to the correct class
    criterion = nn.CrossEntropyLoss()

    # 📝 LEARN: Adam Optimizer
    # Adam = Adaptive Moment Estimation
    # It maintains per-parameter learning rates that adapt based on:
    #   - First moment (mean of gradients)  → like momentum
    #   - Second moment (variance of gradients) → scales LR for each param
    # Generally easier to tune than SGD and converges faster initially.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Step 4: Train! ─────────────────────────────────
    print(f"\n🚀 Step 3: Training for {NUM_EPOCHS} epochs...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        save_path='models/milestone1_best.pth'
    )

    # ── Step 5: Plot Results ───────────────────────────
    print("\n📊 Step 4: Plotting training curves...")
    plot_training_history(
        history,
        title="Milestone 1: Baseline CNN",
        save_path='results/milestone1_curves.png'
    )

    # ── Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📝 Milestone 1 Summary")
    print("=" * 60)
    print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")
    print(f"  Random Guessing:   1.00%")
    print(f"  Beat Random?       {'✅ YES!' if max(history['val_acc']) > 1.0 else '❌ No'}")
    print()
    print("  🔍 Things to Notice:")
    print("    1. Is train accuracy much higher than val accuracy? → Overfitting!")
    print("    2. Is the loss still decreasing? → More epochs might help")
    print("    3. How fast did it converge? → Check the loss curve")
    print()
    print("  ➡️  Next: Milestone 2 — Data Augmentation to fight overfitting!")
    print("=" * 60)


if __name__ == '__main__':
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()
