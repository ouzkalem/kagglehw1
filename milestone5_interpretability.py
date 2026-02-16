"""
🔍 Milestone 5: Interpretability with Captum
=============================================

📝 LEARN: Why Interpretability?
   Deep learning models are often called "black boxes" — we see the input
   and output, but what happens inside is mysterious.

   Interpretability tools help us:
     1. Understand WHAT the model looks at (which pixels matter?)
     2. Build TRUST (is it using the right features?)
     3. DEBUG failures (why did it misclassify this image?)
     4. IMPROVE models (if it looks at backgrounds, we need better augmentation)

📝 LEARN: Occlusion Sensitivity (from Captum)
   The idea is simple:
     1. Slide a small grey patch across the image
     2. At each position, measure how much the prediction changes
     3. If covering a region DROPS the confidence → that region is IMPORTANT
     4. Create a heatmap showing which regions matter most

   This is intuitive: if covering the cat's face makes the model unsure,
   then the model was using the face to make its prediction!
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import CIFAR100_CLASSES, IMAGENET_MEAN, IMAGENET_STD

# We import Captum for interpretability
try:
    from captum.attr import Occlusion, IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    print("⚠️  Captum not installed. Run: pip install captum")
    CAPTUM_AVAILABLE = False


def load_model(model_path, device):
    """Load the best ResNet18 model from Milestone 4."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 100)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


def get_sample_images(num_samples=8):
    """Get sample images from CIFAR-100 validation set."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Get a few samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    return torch.stack(images), labels


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Convert normalized tensor back to displayable image."""
    # 📝 LEARN: We need to reverse the normalization to display images
    # Original: (pixel - mean) / std
    # Reverse:  pixel * std + mean
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)


def visualize_occlusion(model, images, labels, device, save_path=None):
    """
    📝 LEARN: Occlusion Attribution
    ────────────────────────────────
    Captum's Occlusion method:
      1. Creates a sliding window (e.g., 16x16 pixels)
      2. At each position, replaces the window with a baseline (grey)
      3. Measures the change in prediction confidence
      4. Larger change → more important region

    Parameters:
      - sliding_window_shapes: Size of the occluding patch
      - strides: How many pixels to slide between positions
      - baselines: What value to fill the occluded region with (0 = black)
    """
    if not CAPTUM_AVAILABLE:
        print("❌ Captum not available. Install it first!")
        return

    model.eval()
    occlusion = Occlusion(model)

    num_images = min(len(images), 4)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        img = images[i].unsqueeze(0).to(device)
        true_label = labels[i]

        # Get prediction
        with torch.no_grad():
            output = model(img)
            pred_label = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_label].item()

        # Compute occlusion attribution
        # 📝 LEARN: We compute attribution for the PREDICTED class
        # This shows us what the model focuses on when making its prediction
        attribution = occlusion.attribute(
            img,
            target=pred_label,
            sliding_window_shapes=(3, 16, 16),  # 16x16 patch across all 3 channels
            strides=(3, 8, 8),                    # Slide by 8 pixels
            baselines=0                           # Replace with black
        )

        # Display original image
        orig_img = denormalize(images[i]).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original\nTrue: {CIFAR100_CLASSES[true_label]}")
        axes[i, 0].axis('off')

        # Display attribution heatmap
        attr = attribution.squeeze().cpu().detach().numpy()
        attr = np.mean(np.abs(attr), axis=0)  # Average across channels
        axes[i, 1].imshow(attr, cmap='hot')
        axes[i, 1].set_title(f"Occlusion Attribution\nPred: {CIFAR100_CLASSES[pred_label]} ({confidence:.1%})")
        axes[i, 1].axis('off')

        # Overlay attribution on original image
        axes[i, 2].imshow(orig_img)
        axes[i, 2].imshow(attr, cmap='hot', alpha=0.5)
        correct = "✅" if pred_label == true_label else "❌"
        axes[i, 2].set_title(f"Overlay {correct}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("  🔍 Milestone 5: Model Interpretability with Captum")
    print("=" * 60)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'models/milestone4_resnet18_best.pth'

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("   Please run milestone4_transfer_learning.py first!")
        return

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    # Get sample images
    print("\n📷 Getting sample images from CIFAR-100 test set...")
    np.random.seed(42)
    images, labels = get_sample_images(num_samples=4)

    # Run occlusion attribution
    print("\n🔍 Running Occlusion Attribution (this may take a moment)...")
    visualize_occlusion(
        model, images, labels, DEVICE,
        save_path='results/milestone5_occlusion.png'
    )

    print("\n" + "=" * 60)
    print("  📝 Milestone 5 Summary")
    print("=" * 60)
    print("  🔍 Things to look for in the visualizations:")
    print("    1. Does the model focus on the OBJECT or the BACKGROUND?")
    print("    2. For correct predictions: Is it attending to meaningful features?")
    print("    3. For wrong predictions: What confused the model?")
    print()
    print("  ➡️  Next: Milestone 6 — Test on the OOD frog images!")
    print("=" * 60)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main()
