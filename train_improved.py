"""
Improved training script for motion prediction with better generalization.
This script includes best practices for handling diverse datasets like MOT17 and DanceTrack.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

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


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset configuration
    print("Loading datasets...")
    
    # Load MOT17 dataset
    if args.mot17_path:
        train_mot17 = GTSequenceDataset.from_roots(
            root_dirs=[args.mot17_path],
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            seq_total_len=args.seq_total_len,
            random_jump=args.random_jump,
            noise_prob=args.noise_prob,
            noise_coeff=args.noise_coeff,
            random_drop_prob=args.random_drop_prob,
            use_motion_features=args.use_motion_features
        )
        print(f"MOT17 dataset: {len(train_mot17)} samples")
    else:
        train_mot17 = None
    
    # Load MOT20 dataset
    if args.mot20_path:
        train_mot20 = GTSequenceDataset.from_roots(
            root_dirs=[args.mot20_path],
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            seq_total_len=args.seq_total_len,
            random_jump=args.random_jump,
            noise_prob=args.noise_prob,
            noise_coeff=args.noise_coeff,
            random_drop_prob=args.random_drop_prob,
            use_motion_features=args.use_motion_features
        )
        print(f"MOT20 dataset: {len(train_mot20)} samples")
    else:
        train_mot20 = None
    
    # Load DanceTrack dataset
    if args.dancetrack_path:
        train_dancetrack = GTSequenceDataset.from_roots(
            root_dirs=[args.dancetrack_path],
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            seq_total_len=args.seq_total_len,
            random_jump=args.random_jump,
            noise_prob=args.noise_prob,
            noise_coeff=args.noise_coeff,
            random_drop_prob=args.random_drop_prob,
            use_motion_features=args.use_motion_features
        )
        print(f"DanceTrack dataset: {len(train_dancetrack)} samples")
    else:
        train_dancetrack = None
    
    # Combine datasets with balanced sampling
    datasets = [d for d in [train_mot17, train_mot20, train_dancetrack] if d is not None]
    if len(datasets) == 0:
        raise ValueError("At least one dataset path must be provided")
    
    if len(datasets) > 1:
        # Combine datasets
        train_dataset = ConcatDataset(datasets)
        print(f"Combined dataset: {len(train_dataset)} samples")
    else:
        train_dataset = datasets[0]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_loader) * 5,  # Restart every 5 epochs
            T_mult=2,
            eta_min=args.lr * 0.01
        )
    else:
        scheduler = None
    
    # Training loop
    best_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'args': vars(args)
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
    
    print("\nTraining completed!")
    print(f"Best train loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train motion prediction transformer")
    
    # Dataset paths
    parser.add_argument("--mot17_path", type=str, default=None, help="Path to MOT17 dataset")
    parser.add_argument("--mot20_path", type=str, default=None, help="Path to MOT20 dataset")
    parser.add_argument("--dancetrack_path", type=str, default=None, help="Path to DanceTrack dataset")
    
    # Sequence parameters
    parser.add_argument("--seq_in_len", type=int, default=20, help="Input sequence length")
    parser.add_argument("--seq_out_len", type=int, default=10, help="Output sequence length")
    parser.add_argument("--seq_total_len", type=int, default=30, help="Total sequence length")
    parser.add_argument("--random_jump", action="store_true", help="Use random jump augmentation")
    
    # Augmentation
    parser.add_argument("--noise_prob", type=float, default=0.3, help="Probability of adding noise")
    parser.add_argument("--noise_coeff", type=float, default=0.1, help="Noise coefficient")
    parser.add_argument("--random_drop_prob", type=float, default=None, help="Probability of randomly dropping frames (simulating missing detections)")
    
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
    parser.add_argument("--scheduler", type=str, default="onecycle", 
                        choices=["onecycle", "cosine", "none"], help="LR scheduler")
    
    # System
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    main(args)

