"""
📤 Submission Generator — Create Kaggle Submission CSV
======================================================

This script:
1. Loads your best trained model (from Milestone 4)
2. Runs inference on all test images
3. Creates a submission.csv file in the required format:
     filename,class
     test_00000.png,42
     test_00001.png,7
     ...
"""

import sys
import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_test_dataloader, CIFAR100_CLASSES


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
    return model


@torch.no_grad()
def generate_predictions(model, test_loader, device):
    """Run the model on all test images and collect predictions."""
    all_filenames = []
    all_predictions = []

    print("🔮 Generating predictions...")
    for images, filenames in tqdm(test_loader, desc="  Predicting"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_filenames.extend(filenames)
        all_predictions.extend(predicted.cpu().numpy().tolist())

    return all_filenames, all_predictions


def main():
    print("=" * 60)
    print("  📤 Generating Kaggle Submission")
    print("=" * 60)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'models/milestone4_resnet18_best.pth'
    TEST_DIR = 'datas/test_images'
    OUTPUT_PATH = 'submission.csv'

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("   Please run milestone4_transfer_learning.py first!")
        return

    # Check test images exist
    if not os.path.exists(TEST_DIR):
        print(f"\n❌ Test images not found at {TEST_DIR}")
        print("   Please download the competition data and extract test images to data/test_images/")
        return

    # Load model
    print(f"\n📥 Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, DEVICE)

    # Load test data
    print(f"📷 Loading test images from {TEST_DIR}...")
    test_loader = get_test_dataloader(
        test_dir=TEST_DIR,
        batch_size=64,
        image_size=224,           # Must match training size!
        use_imagenet_stats=True   # Must match training normalization!
    )

    # Generate predictions
    filenames, predictions = generate_predictions(model, test_loader, DEVICE)

    # Create submission CSV
    submission = pd.DataFrame({
        'filename': filenames,
        'class': predictions
    })
    submission.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"\n✅ Submission saved to {OUTPUT_PATH}")
    print(f"   Total predictions: {len(submission)}")
    print(f"\n📋 First 10 predictions:")
    for _, row in submission.head(10).iterrows():
        class_name = CIFAR100_CLASSES[row['class']]
        print(f"   {row['filename']} → {row['class']} ({class_name})")

    print(f"\n📊 Class distribution:")
    top_classes = submission['class'].value_counts().head(5)
    for cls_id, count in top_classes.items():
        print(f"   {CIFAR100_CLASSES[cls_id]:20s} (class {cls_id}): {count} predictions")

    print(f"\n🚀 Upload {OUTPUT_PATH} to Kaggle to see your score!")
    print("=" * 60)


if __name__ == '__main__':
    main()
