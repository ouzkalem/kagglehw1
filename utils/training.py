"""
🏋️ training.py — Reusable Training & Evaluation Loops
======================================================

This module provides the core training loop that all milestones share.

📝 LEARN: The Training Loop — The Heart of Deep Learning
   Every neural network learns through the same cycle:
   1. FORWARD PASS:  Input → Model → Prediction
   2. LOSS:          Compare prediction with true label → error number
   3. BACKWARD PASS: Compute gradients (how to adjust each weight)
   4. UPDATE:        Optimizer adjusts weights to reduce the loss
   5. REPEAT for every batch, for many epochs
"""

import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    📝 LEARN: One Epoch = One Complete Pass Through the Training Data
    ─────────────────────────────────────────────────────────────────
    For each batch:
      1. model.train()  → enables dropout, batchnorm in training mode
      2. Forward pass    → images go through the network
      3. criterion()     → compute loss (CrossEntropyLoss for classification)
      4. loss.backward() → compute gradients via backpropagation
      5. optimizer.step()→ update weights using gradients
      6. optimizer.zero_grad() → reset gradients for next batch
    """
    model.train()  # 📝 Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm gives us a nice progress bar
    pbar = tqdm(train_loader, desc="  Training", leave=False)

    for images, labels in pbar:
        # 📝 LEARN: Move data to same device as model (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # Step 1: Forward pass — the model makes predictions
        outputs = model(images)

        # Step 2: Compute loss — how wrong are the predictions?
        # 📝 LEARN: CrossEntropyLoss combines LogSoftmax + NLLLoss
        #    It's the standard loss for multi-class classification.
        #    Lower loss = better predictions.
        loss = criterion(outputs, labels)

        # Step 3: Backward pass — compute gradients
        # 📝 LEARN: This is where the "learning" happens!
        #    PyTorch traverses the computation graph backwards,
        #    computing ∂loss/∂weight for every parameter.
        optimizer.zero_grad()  # Clear old gradients first!
        loss.backward()        # Compute new gradients

        # Step 4: Update weights
        # 📝 LEARN: The optimizer uses gradients to adjust weights.
        #    SGD:  weight -= lr * gradient
        #    Adam: Uses adaptive learning rates + momentum
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.1f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()  # 📝 LEARN: Disables gradient computation → saves memory & speed
def evaluate(model, val_loader, criterion, device):
    """
    📝 LEARN: Evaluation — Testing Without Learning
    ────────────────────────────────────────────────
    During evaluation:
      - model.eval()    → disables dropout, uses running stats for batchnorm
      - torch.no_grad() → no gradient computation needed (faster, less memory)
      - We NEVER call loss.backward() or optimizer.step()
    
    This tells us how well the model generalizes to data it hasn't trained on.
    If train accuracy >> val accuracy, the model is OVERFITTING.
    """
    model.eval()  # 📝 Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="  Validation", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model(
    model, train_loader, val_loader, criterion, optimizer,
    device, num_epochs=10, scheduler=None, save_path='best_model.pth'
):
    """
    📝 LEARN: The Full Training Pipeline
    ─────────────────────────────────────
    This function:
      1. Trains for multiple epochs
      2. Evaluates after each epoch
      3. Saves the BEST model (highest val accuracy)
      4. Optionally adjusts learning rate with a scheduler
      5. Returns training history for plotting
    
    📝 KEY CONCEPT: "Early Stopping" / "Best Model Saving"
       We save the model with the best validation accuracy.
       This prevents us from using an overfitted model —
       training might keep improving, but validation may get worse!
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    start_time = time.time()

    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 📝 LEARN: Learning Rate Scheduler
        # Some schedulers reduce the LR when validation loss stops improving.
        # This helps the model "fine-tune" instead of overshooting.
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved_flag = " ⭐ SAVED (new best!)"
        else:
            saved_flag = ""

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        print(f"  Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% "
              f"| LR: {current_lr:.6f}{saved_flag}")

        # 📝 LEARN: The Overfitting Gap
        # If train_acc is much higher than val_acc, the model is memorizing!
        gap = train_acc - val_acc
        if gap > 20:
            print(f"  ⚠️  Overfitting detected! Gap: {gap:.1f}% — Consider more augmentation or regularization")

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"✅ Training complete in {total_time:.1f}s")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")

    return history


def plot_training_history(history, title="Training History", save_path=None):
    """
    📝 LEARN: Visualizing Training — The Most Important Debugging Tool
    ──────────────────────────────────────────────────────────────────
    By plotting loss and accuracy curves, you can diagnose:
      - UNDERFITTING: Both curves plateau at bad values → need bigger model
      - OVERFITTING:  Train improves but val gets worse → need regularization
      - GOOD FIT:     Both curves improve and converge together → 🎉
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='#FF6B6B', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='#4ECDC4', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='#FF6B6B', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', color='#4ECDC4', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")

    plt.show()
