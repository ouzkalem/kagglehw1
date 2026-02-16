"""
🚀 Milestone 4: Transfer Learning with ResNet18
================================================

📝 LEARN: Transfer Learning — Standing on the Shoulders of Giants
   Training a model from scratch on 50,000 images (CIFAR-100) is hard.
   But what if we could use a model that already learned from 1.2 MILLION
   images (ImageNet)?

   Transfer Learning does exactly this:
   1. Take a pre-trained model (ResNet18 trained on ImageNet)
   2. Replace the final classification layer (1000 classes → 100 classes)
   3. Fine-tune on our specific dataset

   Why does this work?
   - Early layers learn universal features (edges, textures, shapes)
   - These features transfer well across different image tasks
   - We only need to train the model to combine these features
     for OUR specific classes

📝 LEARN: ResNet (Residual Network) — The Architecture
   ResNet solved the "degradation problem": deeper networks performed WORSE
   than shallow ones. The fix? Skip connections (residual connections).

   Instead of learning: H(x) = desired output
   ResNet learns:        F(x) = H(x) - x (the "residual")
   And computes:         H(x) = F(x) + x

   This means the network only needs to learn the DIFFERENCE (residual),
   which is much easier! If a layer isn't helpful, it can learn F(x)=0,
   effectively skipping itself.

   ResNet18 = 18 layers deep with skip connections
   Much deeper than our custom CNN but trains well thanks to residuals!

⚠️ CRITICAL: Pre-trained models expect ImageNet normalization!
   Mean: [0.485, 0.456, 0.406]
   Std:  [0.229, 0.224, 0.225]
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_cifar100_dataloaders
from utils.training import train_model, plot_training_history


def create_resnet18_model(num_classes=100, pretrained=True, freeze_backbone=False):
    """
    📝 LEARN: Modifying a Pre-trained Model
    ────────────────────────────────────────
    ResNet18's original final layer: Linear(512, 1000) → 1000 ImageNet classes
    We replace it with:             Linear(512, 100)  → 100 CIFAR-100 classes

    Two strategies:
    A) Feature Extraction (freeze_backbone=True):
       - Freeze ALL layers except the final one
       - Only train the new classification layer
       - Fast! But limited by pre-learned features

    B) Fine-Tuning (freeze_backbone=False):
       - Train ALL layers (or unfreeze gradually)
       - Slower, but adapts features to our specific task
       - Usually achieves better accuracy
    """
    if pretrained:
        print("  📥 Loading pre-trained ResNet18 (ImageNet weights)...")
        # 📝 LEARN: weights=ResNet18_Weights.IMAGENET1K_V1
        # This loads weights trained on 1.2M ImageNet images across 1000 classes.
        # The model already knows about edges, textures, shapes, and objects!
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        print("  🎲 Creating ResNet18 with RANDOM weights (training from scratch)...")
        model = models.resnet18(weights=None)

    # 📝 LEARN: Freezing Layers
    # When we "freeze" a layer, we set requires_grad=False.
    # This means its weights WON'T be updated during training.
    # Useful for feature extraction: keep the pre-learned features as-is.
    if freeze_backbone:
        print("  🔒 Freezing backbone — only training the final layer")
        for param in model.parameters():
            param.requires_grad = False

    # 📝 LEARN: Replace the Final Layer
    # ResNet18's final layer (fc) maps 512 features → 1000 classes
    # We replace it with our own: 512 → 100
    num_features = model.fc.in_features  # This is 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),                  # Regularization
        nn.Linear(num_features, num_classes)  # New classification layer
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📐 Parameters: {total:,} total, {trainable:,} trainable")

    return model


def main():
    print("=" * 60)
    print("  🚀 Milestone 4: Transfer Learning with ResNet18")
    print("=" * 60)

    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Step 1: Load Data ──────────────────────────────
    # ⚠️ CRITICAL: Use ImageNet normalization for pre-trained models!
    # The model was trained with these specific Mean/Std values.
    # Using CIFAR stats would shift the input distribution and break features.
    print("\n📦 Loading CIFAR-100 with ImageNet normalization...")
    print("   ⚠️  Using ImageNet Mean=[0.485, 0.456, 0.406] Std=[0.229, 0.224, 0.225]")

    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=BATCH_SIZE,
        augment=True,
        image_size=224,           # 📝 ImageNet size! Pre-trained models expect this.
        use_imagenet_stats=True,  # 📝 CRITICAL: Use ImageNet normalization!
    )

    # ── Step 2: Create Pre-trained ResNet18 ────────────
    print("\n🏗️  Creating pre-trained ResNet18...")
    model = create_resnet18_model(
        num_classes=100,
        pretrained=True,
        freeze_backbone=False  # 📝 Fine-tune all layers for best accuracy
    )
    model = model.to(DEVICE)

    # ── Step 3: Optimizer & Scheduler ──────────────────
    # 📝 LEARN: Lower LR for pre-trained models!
    # The pre-trained weights are already "good." We want small adjustments,
    # not big random jumps that would destroy the learned features.
    # Typical: 1e-4 to 5e-4 for fine-tuning (vs 1e-3 for from-scratch)

    # 📝 LEARN: Differential Learning Rates
    # We can use different LRs for different parts of the model:
    #   - Backbone (pre-trained layers): LOW LR (small adjustments)
    #   - Head (new classification layer): HIGHER LR (needs to learn from scratch)
    backbone_params = [p for name, p in model.named_parameters()
                       if 'fc' not in name and p.requires_grad]
    head_params = [p for name, p in model.named_parameters()
                   if 'fc' in name and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},    # Low LR for backbone
        {'params': head_params, 'lr': 1e-3}          # Higher LR for new head
    ], weight_decay=1e-2)

    # ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss()

    # ── Step 4: Train ──────────────────────────────────
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        scheduler=scheduler,
        save_path='models/milestone4_resnet18_best.pth'
    )

    # ── Step 5: Plot ───────────────────────────────────
    plot_training_history(
        history,
        title="Milestone 4: ResNet18 Transfer Learning",
        save_path='results/milestone4_curves.png'
    )

    # ── Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📝 Milestone 4 Summary")
    print("=" * 60)
    print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")
    print()
    print("  🔍 Key Observations:")
    print("    - Pre-trained ResNet18 should significantly outperform custom CNN")
    print("    - The model converges MUCH faster (features already learned!)")
    print("    - ImageNet normalization is CRITICAL — wrong stats = bad results")
    print()
    print("  ➡️  Next: Run generate_submission.py to create your Kaggle submission!")
    print("  ➡️  Then: Milestone 5 — Interpretability with Captum")
    print("=" * 60)


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()
