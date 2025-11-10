"""
Improved training script for motion prediction with better generalization.
This script includes best practices for handling diverse datasets like MOT17 and DanceTrack.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import time

from dataset import GTSequenceDataset
from transformer_encoder import MotionTransformer
from loss import LossFunction


def compute_pixel_error(preds, targets, image_sizes):
    """
    Compute pixel error by denormalizing predictions.
    
    Args:
        preds: (B, T, 13) or (B, T, 5) normalized predictions
        targets: (B, T, 13) or (B, T, 5) normalized ground truth
        image_sizes: list of (width, height) tuples for each batch item
    """
    pred_boxes = preds[:, :, :4].detach().cpu().numpy()
    target_boxes = targets[:, :, :4].detach().cpu().numpy()
    
    errors = []
    for i, (w, h) in enumerate(image_sizes):
        scale = np.array([w, h, w, h])
        pred_px = pred_boxes[i] * scale
        target_px = target_boxes[i] * scale
        # Compute center point error
        pred_center = (pred_px[:, :2] + pred_px[:, 2:]) / 2
        target_center = (target_px[:, :2] + target_px[:, 2:]) / 2
        error = np.linalg.norm(pred_center - target_center, axis=1).mean()
        errors.append(error)
    
    return np.mean(errors)


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        src, trg, gt_src, gt_trg = batch
        src = src.to(device)
        trg = trg.to(device)
        gt_trg = gt_trg.to(device)
        
        optimizer.zero_grad()
        
        # Use teacher forcing during training
        output = model(src, trg[:, :-1])
        
        # Compute loss against ground truth
        loss = criterion(gt_trg[:, 1:], output)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * src.size(0)
        total_samples += src.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, dataset_name=""):
    model.eval()
    total_loss = 0
    total_samples = 0
    total_pixel_error = 0
    
    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        src, trg, gt_src, gt_trg = batch
        src = src.to(device)
        trg = trg.to(device)
        gt_trg = gt_trg.to(device)
        
        # Autoregressive inference
        output = model.inference(src, trg, num_steps=trg.size(1) - 1)
        
        loss = criterion(gt_trg[:, 1:], output)
        
        total_loss += loss.item() * src.size(0)
        total_samples += src.size(0)
        
        # Compute pixel error (assuming normalized coordinates)
        # For proper evaluation, you'd need to pass image dimensions
        pred_boxes = output[:, :, :4]
        target_boxes = gt_trg[:, 1:, :4]
        pixel_error = torch.norm(pred_boxes - target_boxes, dim=-1).mean().item()
        total_pixel_error += pixel_error * src.size(0)
    
    avg_loss = total_loss / total_samples
    avg_pixel_error = total_pixel_error / total_samples
    
    print(f"{dataset_name} - Loss: {avg_loss:.4f}, Normalized Pixel Error: {avg_pixel_error:.4f}")
    
    return avg_loss, avg_pixel_error


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset configuration
    print("Loading datasets...")
    print("="*60)
    
    # Collect train dataset paths
    train_paths = []
    if args.mot17_train_path:
        train_paths.append(args.mot17_train_path)
        print(f"MOT17 train: {args.mot17_train_path}")
    if args.mot20_train_path:
        train_paths.append(args.mot20_train_path)
        print(f"MOT20 train: {args.mot20_train_path}")
    if args.dancetrack_train_path:
        train_paths.append(args.dancetrack_train_path)
        print(f"DanceTrack train: {args.dancetrack_train_path}")
    
    if len(train_paths) == 0:
        raise ValueError("At least one training dataset path must be provided")
    
    # Load training dataset
    print("\nLoading training data...")
    train_dataset = GTSequenceDataset.from_roots(
        root_dirs=train_paths,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        seq_total_len=args.seq_total_len,
        random_jump=args.random_jump,
        noise_prob=args.noise_prob,
        noise_coeff=args.noise_coeff,
        random_drop_prob=args.random_drop_prob,
        use_motion_features=args.use_motion_features
    )
    print(f"Training set: {len(train_dataset)} samples")
    
    # Collect validation dataset paths
    val_paths = []
    if args.mot17_val_path:
        val_paths.append(args.mot17_val_path)
        print(f"\nMOT17 val: {args.mot17_val_path}")
    if args.mot20_val_path:
        val_paths.append(args.mot20_val_path)
        print(f"MOT20 val: {args.mot20_val_path}")
    if args.dancetrack_val_path:
        val_paths.append(args.dancetrack_val_path)
        print(f"DanceTrack val: {args.dancetrack_val_path}")
    
    if len(val_paths) == 0:
        raise ValueError("At least one validation dataset path must be provided")
    
    # Load validation dataset (with optional different augmentation)
    print("\nLoading validation data...")
    val_dataset = GTSequenceDataset.from_roots(
        root_dirs=val_paths,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        seq_total_len=args.seq_total_len,
        random_jump=False,  # No random jump for validation
        noise_prob=args.val_noise_prob if args.val_noise_prob is not None else None,
        noise_coeff=args.val_noise_coeff if args.val_noise_prob is not None else None,
        random_drop_prob=None,  # No random drop for validation
        use_motion_features=args.use_motion_features
    )
    print(f"Validation set: {len(val_dataset)} samples")
    print("="*60)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model configuration
    input_dim = 13 if args.use_motion_features else 5
    model = MotionTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        use_residual_prediction=args.use_residual
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = LossFunction(
        loss1_coeff=args.loss1_coeff,
        loss2_coeff=args.loss2_coeff,
        loss3_coeff=args.loss3_coeff,
        loss4_coeff=args.loss4_coeff,
        use_motion_features=args.use_motion_features
    )
    
    # Optimizer with weight decay for regularization
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # Learning rate scheduler
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        scheduler_type = 'step'  # Update per batch
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_loader) * 5,  # Restart every 5 epochs
            T_mult=2,
            eta_min=args.lr * 0.01
        )
        scheduler_type = 'step'  # Update per batch
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.patience // 2,  # Reduce LR before early stopping
            verbose=True,
            min_lr=args.lr * 0.001
        )
        scheduler_type = 'epoch'  # Update per epoch based on validation
    else:
        scheduler = None
        scheduler_type = None
    
    # Training loop with validation and early stopping
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    patience_counter = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'epochs': []
    }
    
    print("\nStarting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Learning rate scheduler: {args.scheduler}")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Training phase
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            scheduler if scheduler_type == 'step' else None
        )
        
        # Validation phase
        val_loss, val_pixel_error = evaluate(
            model, val_loader, criterion, device, "Validation"
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate based on validation loss (if using plateau scheduler)
        if scheduler_type == 'epoch' and scheduler is not None:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        history['epochs'].append(epoch + 1)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  Val Pixel Error: {val_pixel_error:.4f}")
        print(f"  Learning Rate:   {current_lr:.2e}")
        print(f"  Epoch Time:      {epoch_time:.1f}s")
        print(f"{'='*60}")
        
        # Check if this is the best model based on VALIDATION loss
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            patience_counter = 0
            improved = True
            
            # Save best model
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_pixel_error': val_pixel_error,
                'args': vars(args)
            }, best_model_path)
            print(f"✓ New best model saved! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"✗ No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_pixel_error': val_pixel_error,
                'history': history,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"No improvement for {args.patience} consecutive epochs")
            print(f"{'='*60}")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f} (at epoch {history['val_loss'].index(best_val_loss) + 1})")
    print(f"Corresponding train loss: {best_train_loss:.4f}")
    print(f"Best model saved to: {save_dir / 'best_model.pth'}")
    print("="*60)
    
    # Save training history
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = history['epochs']
        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        ax1.axvline(x=history['val_loss'].index(min(history['val_loss'])) + 1, 
                   color='g', linestyle='--', label='Best Model', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, history['learning_rates'], marker='o', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        plt.close()
    except ImportError:
        print("Note: Install matplotlib to automatically generate training curve plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train motion prediction transformer")
    
    # Training dataset paths
    parser.add_argument("--mot17_train_path", type=str, default=None, 
                        help="Path to MOT17 train dataset (e.g., /path/to/MOT17/train)")
    parser.add_argument("--mot20_train_path", type=str, default=None, 
                        help="Path to MOT20 train dataset (e.g., /path/to/MOT20/train)")
    parser.add_argument("--dancetrack_train_path", type=str, default=None, 
                        help="Path to DanceTrack train dataset (e.g., /path/to/DanceTrack/train)")
    
    # Validation dataset paths
    parser.add_argument("--mot17_val_path", type=str, default=None, 
                        help="Path to MOT17 val dataset (e.g., /path/to/MOT17/val)")
    parser.add_argument("--mot20_val_path", type=str, default=None, 
                        help="Path to MOT20 val dataset (e.g., /path/to/MOT20/val)")
    parser.add_argument("--dancetrack_val_path", type=str, default=None, 
                        help="Path to DanceTrack val dataset (e.g., /path/to/DanceTrack/val)")
    
    # Sequence parameters
    parser.add_argument("--seq_in_len", type=int, default=20, help="Input sequence length")
    parser.add_argument("--seq_out_len", type=int, default=10, help="Output sequence length")
    parser.add_argument("--seq_total_len", type=int, default=30, help="Total sequence length")
    parser.add_argument("--random_jump", action="store_true", help="Use random jump augmentation")
    
    # Training augmentation
    parser.add_argument("--noise_prob", type=float, default=0.3, help="Probability of adding noise to training data")
    parser.add_argument("--noise_coeff", type=float, default=0.1, help="Noise coefficient for training data")
    parser.add_argument("--random_drop_prob", type=float, default=None, 
                        help="Probability of randomly dropping frames in training (simulating missing detections)")
    
    # Validation augmentation (optional, usually less aggressive)
    parser.add_argument("--val_noise_prob", type=float, default=None, 
                        help="Probability of adding noise to validation data (default: None = no noise)")
    parser.add_argument("--val_noise_coeff", type=float, default=0.5, 
                        help="Noise coefficient for validation data (if val_noise_prob is set)")
    
    # Model architecture
    parser.add_argument("--use_motion_features", action="store_true", default=True, 
                        help="Use velocity and acceleration features")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dim_ff", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_residual", action="store_true", default=True,
                        help="Use residual prediction from last known position")
    
    # Loss coefficients
    parser.add_argument("--loss1_coeff", type=float, default=1.0, help="IOU loss coefficient")
    parser.add_argument("--loss2_coeff", type=float, default=1.0, help="Bbox loss coefficient")
    parser.add_argument("--loss3_coeff", type=float, default=0.5, help="Confidence loss coefficient")
    parser.add_argument("--loss4_coeff", type=float, default=0.5, help="Motion features loss coefficient")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="plateau", 
                        choices=["onecycle", "cosine", "plateau", "none"], 
                        help="LR scheduler (plateau recommended for validation-based training)")
    
    # Validation and early stopping
    parser.add_argument("--patience", type=int, default=20, 
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # System
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    main(args)

