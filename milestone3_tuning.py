"""
⚙️ Milestone 3: The Art of Tuning — Training Dynamics
======================================================

📝 LEARN: Hyperparameters — The Knobs You Turn
   Model parameters (weights) are learned during training.
   HYPERparameters are set BY YOU and control HOW the model learns:
     - Learning Rate: How big are the weight updates?
     - Optimizer: What strategy for updating weights?
     - Batch Size: How many images per gradient update?
     - Scheduler: How does the learning rate change over time?
     - Weight Decay: How much do we penalize large weights?

   Finding good hyperparameters is more art than science!
   This script lets you experiment with different configurations.
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_cifar100_dataloaders
from utils.training import train_model, plot_training_history
from milestone2_augmentation import AugmentedCNN


def run_experiment(name, model, train_loader, val_loader, optimizer, device,
                   num_epochs=15, scheduler=None):
    """Run a single experiment and return history."""
    print(f"\n{'=' * 60}")
    print(f"  ⚙️  Experiment: {name}")
    print(f"{'=' * 60}")

    criterion = nn.CrossEntropyLoss()
    save_path = f'models/milestone3_{name.lower().replace(" ", "_")}.pth'

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=num_epochs,
        scheduler=scheduler,
        save_path=save_path
    )
    return history


# ============================================================================
# 📝 EXPERIMENTS
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def experiment_optimizer_comparison():
    """
    📝 LEARN: SGD vs Adam — Two Philosophies of Optimization
    ─────────────────────────────────────────────────────────
    SGD (Stochastic Gradient Descent):
      weight -= lr * gradient
      Simple! But needs careful LR tuning and momentum.
      Often generalizes BETTER in the long run.

    Adam (Adaptive Moment Estimation):
      Maintains per-parameter adaptive learning rates.
      Converges faster initially, but may generalize worse.
      Uses running averages of gradients (momentum) and squared
      gradients (scaling) to adapt each parameter individually.

    Rule of thumb:
      - Adam for quick experiments and prototyping
      - SGD + momentum + scheduler for final competitive models
    """
    print("\n" + "🔬" * 30)
    print("  EXPERIMENT: SGD vs Adam")
    print("🔬" * 30)

    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=64, augment=True, image_size=32
    )

    # Adam experiment
    model_adam = AugmentedCNN(num_classes=100).to(DEVICE)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)
    history_adam = run_experiment("Adam lr=1e-3", model_adam, train_loader, val_loader,
                                  opt_adam, DEVICE, num_epochs=15)

    # SGD + Momentum experiment
    model_sgd = AugmentedCNN(num_classes=100).to(DEVICE)
    opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
    history_sgd = run_experiment("SGD lr=0.01 momentum=0.9", model_sgd, train_loader,
                                 val_loader, opt_sgd, DEVICE, num_epochs=15)

    print(f"\n📊 Results:")
    print(f"   Adam  best val acc: {max(history_adam['val_acc']):.2f}%")
    print(f"   SGD   best val acc: {max(history_sgd['val_acc']):.2f}%")

    return history_adam, history_sgd


def experiment_scheduler():
    """
    📝 LEARN: Learning Rate Schedulers — Dynamic Learning Rates
    ────────────────────────────────────────────────────────────
    A fixed learning rate is rarely optimal:
      - Start high: big steps to get to a good region quickly
      - End low: small steps to fine-tune within that region

    ReduceLROnPlateau:
      Monitors a metric (e.g., val_loss).
      If it doesn't improve for 'patience' epochs, reduce LR by a factor.
      Example: patience=3, factor=0.5 → if loss doesn't improve for 3 epochs,
               multiply LR by 0.5

    CosineAnnealingLR:
      LR follows a cosine curve from max → min.
      Smooth and gradual reduction.
    """
    print("\n" + "🔬" * 30)
    print("  EXPERIMENT: With vs Without Scheduler")
    print("🔬" * 30)

    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=64, augment=True, image_size=32
    )

    # With ReduceLROnPlateau
    model = AugmentedCNN(num_classes=100).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 📝 LEARN: ReduceLROnPlateau parameters
    #   mode='min'   → we want to minimize val_loss
    #   factor=0.5   → multiply LR by 0.5 when triggered
    #   patience=3   → wait 3 epochs before reducing
    #   verbose=True → print when LR changes
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    history = run_experiment("Adam + ReduceLROnPlateau", model, train_loader,
                              val_loader, optimizer, DEVICE, num_epochs=20,
                              scheduler=scheduler)

    print(f"\n📊 Best val acc with scheduler: {max(history['val_acc']):.2f}%")
    return history


def experiment_weight_decay():
    """
    📝 LEARN: Weight Decay (L2 Regularization)
    ───────────────────────────────────────────
    Weight decay adds a penalty for large weights:
      loss = original_loss + (weight_decay/2) * sum(weights²)

    This encourages the model to use small weight values, which:
      - Prevents any single feature from dominating
      - Creates smoother decision boundaries
      - Reduces overfitting

    AdamW is preferred over Adam + weight_decay because it applies
    weight decay correctly (decoupled from gradients).
    """
    print("\n" + "🔬" * 30)
    print("  EXPERIMENT: Weight Decay Effect")
    print("🔬" * 30)

    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=64, augment=True, image_size=32
    )

    model = AugmentedCNN(num_classes=100).to(DEVICE)
    # 📝 LEARN: AdamW — Adam with Decoupled Weight Decay
    # Regular Adam applies weight decay as part of the gradient,
    # which interferes with the adaptive learning rate mechanism.
    # AdamW fixes this by applying weight decay separately.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    history = run_experiment("AdamW + weight_decay=1e-2", model, train_loader,
                              val_loader, optimizer, DEVICE, num_epochs=20,
                              scheduler=scheduler)

    print(f"\n📊 Best val acc with weight decay: {max(history['val_acc']):.2f}%")
    return history


def main():
    print("=" * 60)
    print("  ⚙️  Milestone 3: Training Dynamics Experiments")
    print("=" * 60)
    print()
    print("  📝 This milestone runs experiments to understand how")
    print("     hyperparameters affect training. Each experiment")
    print("     tests a different configuration.")
    print()
    print("  Choose an experiment to run:")
    print("    1. Optimizer Comparison (SGD vs Adam)")
    print("    2. Learning Rate Scheduler")
    print("    3. Weight Decay (L2 Regularization)")
    print("    4. Run ALL experiments")

    choice = input("\n  Enter choice (1-4): ").strip()

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    if choice == '1':
        experiment_optimizer_comparison()
    elif choice == '2':
        experiment_scheduler()
    elif choice == '3':
        experiment_weight_decay()
    elif choice == '4':
        experiment_optimizer_comparison()
        experiment_scheduler()
        experiment_weight_decay()
    else:
        print("  Invalid choice. Running scheduler experiment by default.")
        experiment_scheduler()


if __name__ == '__main__':
    main()
