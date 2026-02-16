"""
🐸 Milestone 6: Out-of-Distribution (OOD) Testing
===================================================

📝 LEARN: What is OOD (Out-of-Distribution)?
   Your model was trained on CIFAR-100 images: small, centered, clean.
   But real-world images are different: different sizes, angles, backgrounds.

   OOD testing reveals how robust your model really is:
     - Does it rely on specific spatial positions? (positional bias)
     - Does it look at the right features? (semantic understanding)
     - How confident is it on images it's never seen? (calibration)

   The frog images are NOT from CIFAR-100 — they're designed to challenge
   your model's understanding of "frog" vs the visual tricks in the image.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import CIFAR100_CLASSES, IMAGENET_MEAN, IMAGENET_STD

try:
    from captum.attr import Occlusion
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


def load_model(model_path, device):
    """Load the best ResNet18 model."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 100)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_and_preprocess(image_path, image_size=224):
    """Load a single image and apply the same preprocessing as training."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, tensor


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse normalization for display."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)


def analyze_image(model, image_path, device, ax_row=None):
    """
    📝 LEARN: Full Analysis Pipeline for a Single Image
    ────────────────────────────────────────────────────
    1. Load and preprocess the image
    2. Get the model's prediction and top-5 classes
    3. (Optional) Run Captum occlusion analysis
    4. Visualize everything
    """
    orig_image, tensor = load_and_preprocess(image_path)
    tensor = tensor.to(device)

    # Get prediction
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    # Top 5 predictions
    top5_probs, top5_indices = probs.topk(5)

    print(f"\n  📷 Image: {os.path.basename(image_path)}")
    print(f"  Top 5 Predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        marker = "👑" if i == 0 else "  "
        print(f"    {marker} {i+1}. {CIFAR100_CLASSES[idx]:20s} → {prob.item():.1%}")

    if ax_row is not None:
        # Show original image
        ax_row[0].imshow(orig_image)
        ax_row[0].set_title(f"Input: {os.path.basename(image_path)}", fontsize=10)
        ax_row[0].axis('off')

        # Show top-5 bar chart
        classes = [CIFAR100_CLASSES[idx] for idx in top5_indices.cpu()]
        values = top5_probs.cpu().numpy()
        colors = ['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(5)]
        ax_row[1].barh(classes[::-1], values[::-1], color=colors[::-1])
        ax_row[1].set_xlabel('Confidence')
        ax_row[1].set_title('Top-5 Predictions', fontsize=10)
        ax_row[1].set_xlim(0, 1)

        # Captum occlusion
        if CAPTUM_AVAILABLE:
            occlusion = Occlusion(model)
            pred_label = top5_indices[0].item()
            attribution = occlusion.attribute(
                tensor, target=pred_label,
                sliding_window_shapes=(3, 16, 16),
                strides=(3, 8, 8), baselines=0
            )
            attr = attribution.squeeze().cpu().detach().numpy()
            attr = np.mean(np.abs(attr), axis=0)

            display_img = denormalize(tensor.squeeze().cpu()).permute(1, 2, 0).numpy()
            ax_row[2].imshow(display_img)
            ax_row[2].imshow(attr, cmap='hot', alpha=0.5)
            ax_row[2].set_title(f'Captum: What model sees\n({CIFAR100_CLASSES[pred_label]})',
                               fontsize=10)
            ax_row[2].axis('off')

    return top5_indices[0].item(), top5_probs[0].item()


def main():
    print("=" * 60)
    print("  🐸 Milestone 6: OOD Testing with Frog Images")
    print("=" * 60)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'models/milestone4_resnet18_best.pth'

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("   Please run milestone4_transfer_learning.py first!")
        return

    model = load_model(MODEL_PATH, DEVICE)

    # Check for frog images
    frog_images = []
    for fname in ['frog.png', 'frog-flip.png']:
        # Check in data/ directory and current directory
        for dir_path in ['datas', 'data', '.']:
            fpath = os.path.join(dir_path, fname)
            if os.path.exists(fpath):
                frog_images.append(fpath)
                break
        else:
            print(f"  ⚠️  {fname} not found in data/ or current directory")

    if not frog_images:
        print("\n❌ No frog images found!")
        print("   Download frog.png and frog-flip.png from the competition")
        print("   and place them in the data/ folder.")
        return

    # Analyze each frog image
    num_images = len(frog_images)
    num_cols = 3 if CAPTUM_AVAILABLE else 2
    fig, axes = plt.subplots(num_images, num_cols, figsize=(5 * num_cols, 5 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    predictions = []
    for i, img_path in enumerate(frog_images):
        pred_class, confidence = analyze_image(model, img_path, DEVICE, axes[i])
        predictions.append((os.path.basename(img_path), pred_class, confidence))

    plt.tight_layout()
    plt.savefig('results/milestone6_ood_frogs.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to results/milestone6_ood_frogs.png")
    plt.show()

    # ── Invariance Check ───────────────────────────────
    if len(predictions) >= 2:
        print("\n" + "=" * 60)
        print("  🔄 Invariance Check")
        print("=" * 60)
        p1_name, p1_class, p1_conf = predictions[0]
        p2_name, p2_class, p2_conf = predictions[1]

        print(f"  {p1_name}: {CIFAR100_CLASSES[p1_class]} ({p1_conf:.1%})")
        print(f"  {p2_name}: {CIFAR100_CLASSES[p2_class]} ({p2_conf:.1%})")

        if p1_class == p2_class:
            print(f"\n  ✅ Same prediction! Model shows spatial invariance.")
        else:
            print(f"\n  ⚠️  Different predictions!")
            print(f"     The flipped version changed the prediction,")
            print(f"     suggesting the model relies on spatial positioning")
            print(f"     rather than semantic features.")

    print("\n" + "=" * 60)
    print("  📝 Milestone 6 Summary")
    print("=" * 60)
    print("  🔍 Questions to consider:")
    print("    1. Did the model predict 'frog' for either image?")
    print("    2. Is the flipped version's prediction the same?")
    print("    3. What does the Captum heatmap reveal?")
    print("    4. Share your results in the 'First Blood' thread!")
    print("=" * 60)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main()
