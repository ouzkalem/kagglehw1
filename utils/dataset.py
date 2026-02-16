"""
📦 dataset.py — Data Loading Utilities for CIFAR-100
=====================================================

This module handles everything related to loading and preparing data.

📝 LEARN: Why do we need a separate dataset module?
   - Keeps data loading logic DRY (Don't Repeat Yourself)
   - All milestones share the same data pipeline
   - Easy to swap transforms without rewriting data loading code
"""

import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# ============================================================================
# 📝 LEARN: CIFAR-100 Class Names
# ============================================================================
# CIFAR-100 has 100 fine-grained classes organized into 20 superclasses.
# For example, superclass "fish" contains: aquarium_fish, flatfish, ray, shark, trout
# This hierarchy can help models learn related concepts!

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# 📝 LEARN: CIFAR-100 Superclasses (20 coarse categories)
# Each superclass groups 5 fine-grained classes
CIFAR100_SUPERCLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


# ============================================================================
# 📝 LEARN: Normalization — Why do we normalize images?
# ============================================================================
# Neural networks train better when input values are small and centered around 0.
# Raw pixel values are 0-255, which creates very large gradients.
#
# Two common normalization strategies:
#   1. CIFAR-100 stats: Computed from the CIFAR-100 training set itself
#   2. ImageNet stats: Used when loading pre-trained models (ResNet, etc.)
#
# ⚠️ CRITICAL: When using pre-trained models, you MUST use ImageNet stats!
#    The model learned features assuming these specific value ranges.

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_cifar100_transforms(augment=False, image_size=32, use_imagenet_stats=False):
    """
    📝 LEARN: What are transforms?
    Transforms are applied to each image before feeding it to the model.
    They can:
      - Resize images (Resize)
      - Convert to tensor (ToTensor) — changes pixels from 0-255 to 0.0-1.0
      - Normalize (Normalize) — centers values around 0 with std=1
      - Augment data (RandomFlip, etc.) — only during training!

    Args:
        augment: If True, apply data augmentation (only for training!)
        image_size: Target size (32 for CIFAR, 224 for ImageNet-style models)
        use_imagenet_stats: If True, use ImageNet normalization (for pre-trained models)
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else CIFAR100_MEAN
    std = IMAGENET_STD if use_imagenet_stats else CIFAR100_STD

    if augment:
        # 📝 LEARN: Data Augmentation
        # Each time we load an image, we apply RANDOM transformations.
        # This means the model sees slightly different versions each epoch,
        # effectively "expanding" our dataset and reducing overfitting.
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),     # 50% chance to flip horizontally
            transforms.RandomCrop(image_size, padding=4), # Randomly shift the crop
            transforms.ColorJitter(                       # Randomly change brightness/contrast
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(15),                # Rotate up to ±15 degrees
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    else:
        # 📝 LEARN: No augmentation for validation/test!
        # We want consistent, reproducible results when evaluating.
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    return transforms.Compose(transform_list)


def get_cifar100_dataloaders(
    data_dir='./data',
    batch_size=64,
    augment=False,
    image_size=32,
    use_imagenet_stats=False,
    val_split=0.1,
    num_workers=0
):
    """
    📝 LEARN: DataLoaders — How PyTorch feeds batches to the model
    ─────────────────────────────────────────────────────────────
    Instead of loading ALL images into memory and processing them one-by-one,
    DataLoader:
      1. Loads a "batch" of images at a time (e.g., 64 images)
      2. Shuffles the training data each epoch (prevents learning order patterns)
      3. Uses multiple CPU threads to load data in parallel

    We also split the training set into train/val:
      - Training set: Used to update model weights (learn)
      - Validation set: Used to check if the model is overfitting

    Args:
        data_dir: Where to download/store CIFAR-100
        batch_size: Number of images per batch
        augment: Whether to apply data augmentation to training data
        image_size: Target image size
        use_imagenet_stats: Whether to use ImageNet normalization
        val_split: Fraction of training data to use for validation
        num_workers: Number of parallel data loading threads
    """
    # Create transforms
    train_transform = get_cifar100_transforms(
        augment=augment, image_size=image_size, use_imagenet_stats=use_imagenet_stats
    )
    val_transform = get_cifar100_transforms(
        augment=False, image_size=image_size, use_imagenet_stats=use_imagenet_stats
    )

    # Download CIFAR-100 (only downloads the first time)
    full_train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=val_transform
    )

    # 📝 LEARN: Train/Val Split
    # We take 10% of training data as validation to monitor overfitting.
    # We use the SAME random split every time (seed=42) for reproducibility.
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(val_split * num_train))

    train_indices = indices[split:]
    val_indices = indices[:split]

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"📊 Dataset loaded:")
    print(f"   Training samples:   {len(train_subset)}")
    print(f"   Validation samples: {len(val_subset)}")
    print(f"   Batch size:         {batch_size}")
    print(f"   Image size:         {image_size}x{image_size}")
    print(f"   Augmentation:       {'✅ Yes' if augment else '❌ No'}")
    print(f"   Normalization:      {'ImageNet' if use_imagenet_stats else 'CIFAR-100'}")

    return train_loader, val_loader


class CompetitionTestDataset(Dataset):
    """
    📝 LEARN: Custom Dataset
    ────────────────────────
    PyTorch's Dataset class requires two methods:
      - __len__: Returns the total number of samples
      - __getitem__: Returns one sample (image, label) by index

    For the competition test set, we don't have labels — we just load
    images from the provided folder and return the filenames.
    """
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        # Get sorted list of image files
        self.image_files = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_test_dataloader(test_dir, batch_size=64, image_size=32, use_imagenet_stats=False):
    """Load the competition test images for generating predictions."""
    transform = get_cifar100_transforms(
        augment=False, image_size=image_size, use_imagenet_stats=use_imagenet_stats
    )
    dataset = CompetitionTestDataset(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"🧪 Test set loaded: {len(dataset)} images")
    return loader
