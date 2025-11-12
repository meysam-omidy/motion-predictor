"""
Visualization script to debug motion prediction errors.
Helps understand why DanceTrack has higher errors than MOT17.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from dataset import GTSequenceDataset
from transformer_encoder import MotionTransformer


def plot_trajectory(ax, trajectory, label, color, marker='o'):
    """Plot a single trajectory (sequence of bboxes)."""
    # Extract center points
    centers = np.zeros((len(trajectory), 2))
    centers[:, 0] = (trajectory[:, 0] + trajectory[:, 2]) / 2  # center x
    centers[:, 1] = (trajectory[:, 1] + trajectory[:, 3]) / 2  # center y
    
    ax.plot(centers[:, 0], centers[:, 1], 
            marker=marker, label=label, color=color, linewidth=2, markersize=6)
    
    # Mark start and end
    ax.scatter(centers[0, 0], centers[0, 1], 
              s=200, marker='*', color=color, edgecolors='black', linewidth=2, zorder=10)
    ax.scatter(centers[-1, 0], centers[-1, 1], 
              s=200, marker='s', color=color, edgecolors='black', linewidth=2, zorder=10)


def plot_velocity_profile(ax, trajectory, label, color):
    """Plot velocity magnitude over time."""
    centers = np.zeros((len(trajectory), 2))
    centers[:, 0] = (trajectory[:, 0] + trajectory[:, 2]) / 2
    centers[:, 1] = (trajectory[:, 1] + trajectory[:, 3]) / 2
    
    velocities = np.diff(centers, axis=0)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    ax.plot(velocity_magnitudes, marker='o', label=label, color=color, linewidth=2)


def plot_acceleration_profile(ax, trajectory, label, color):
    """Plot acceleration magnitude over time."""
    centers = np.zeros((len(trajectory), 2))
    centers[:, 0] = (trajectory[:, 0] + trajectory[:, 2]) / 2
    centers[:, 1] = (trajectory[:, 1] + trajectory[:, 3]) / 2
    
    velocities = np.diff(centers, axis=0)
    accelerations = np.diff(velocities, axis=0)
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    ax.plot(acceleration_magnitudes, marker='o', label=label, color=color, linewidth=2)


def analyze_motion_characteristics(dataset, dataset_name, num_samples=100):
    """Analyze motion characteristics of a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*60}")
    
    velocities = []
    accelerations = []
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        src, trg, gt_src, gt_trg = dataset[idx]
        
        # Combine source and target
        full_seq = torch.cat([gt_src, gt_trg], dim=0).numpy()
        boxes = full_seq[:, :4]
        
        # Compute centers
        centers = np.zeros((len(boxes), 2))
        centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        
        # Compute velocity
        vel = np.diff(centers, axis=0)
        vel_mag = np.linalg.norm(vel, axis=1)
        velocities.extend(vel_mag)
        
        # Compute acceleration
        acc = np.diff(vel, axis=0)
        acc_mag = np.linalg.norm(acc, axis=1)
        accelerations.extend(acc_mag)
    
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    print(f"\nVelocity Statistics:")
    print(f"  Mean: {velocities.mean():.6f}")
    print(f"  Std:  {velocities.std():.6f}")
    print(f"  Min:  {velocities.min():.6f}")
    print(f"  Max:  {velocities.max():.6f}")
    print(f"  Median: {np.median(velocities):.6f}")
    print(f"  95th percentile: {np.percentile(velocities, 95):.6f}")
    
    print(f"\nAcceleration Statistics:")
    print(f"  Mean: {accelerations.mean():.6f}")
    print(f"  Std:  {accelerations.std():.6f}")
    print(f"  Min:  {accelerations.min():.6f}")
    print(f"  Max:  {accelerations.max():.6f}")
    print(f"  Median: {np.median(accelerations):.6f}")
    print(f"  95th percentile: {np.percentile(accelerations, 95):.6f}")
    
    return velocities, accelerations


def visualize_sample_predictions(model, dataset, device, num_samples=5, save_dir='visualizations'):
    """Visualize model predictions vs ground truth."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            src, trg, gt_src, gt_trg = dataset[idx]
            
            # Add batch dimension
            src_batch = src.unsqueeze(0).to(device)
            trg_batch = trg.unsqueeze(0).to(device)
            
            # Get predictions
            pred = model.inference(src_batch, trg_batch, num_steps=trg.size(0) - 1)
            pred = pred.squeeze(0).cpu().numpy()
            
            # Convert to numpy
            src_np = src.numpy()
            gt_trg_np = gt_trg.numpy()
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Trajectory comparison
            ax = axes[0, 0]
            plot_trajectory(ax, src_np[:, :4], 'Input', 'blue')
            plot_trajectory(ax, gt_trg_np[1:, :4], 'Ground Truth', 'green', marker='s')
            plot_trajectory(ax, pred[:, :4], 'Prediction', 'red', marker='^')
            ax.set_xlabel('X (normalized)')
            ax.set_ylabel('Y (normalized)')
            ax.set_title('Trajectory Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Plot 2: Velocity comparison
            ax = axes[0, 1]
            full_gt = np.concatenate([src_np[:, :4], gt_trg_np[1:, :4]], axis=0)
            full_pred = np.concatenate([src_np[:, :4], pred[:, :4]], axis=0)
            plot_velocity_profile(ax, full_gt, 'Ground Truth', 'green')
            plot_velocity_profile(ax, full_pred, 'Prediction', 'red')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Velocity Magnitude')
            ax.set_title('Velocity Profile')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Acceleration comparison
            ax = axes[1, 0]
            plot_acceleration_profile(ax, full_gt, 'Ground Truth', 'green')
            plot_acceleration_profile(ax, full_pred, 'Prediction', 'red')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Acceleration Magnitude')
            ax.set_title('Acceleration Profile')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Error over time
            ax = axes[1, 1]
            gt_centers = np.zeros((len(gt_trg_np[1:]), 2))
            gt_centers[:, 0] = (gt_trg_np[1:, 0] + gt_trg_np[1:, 2]) / 2
            gt_centers[:, 1] = (gt_trg_np[1:, 1] + gt_trg_np[1:, 3]) / 2
            
            pred_centers = np.zeros((len(pred), 2))
            pred_centers[:, 0] = (pred[:, 0] + pred[:, 2]) / 2
            pred_centers[:, 1] = (pred[:, 1] + pred[:, 3]) / 2
            
            errors = np.linalg.norm(gt_centers - pred_centers, axis=1)
            ax.plot(errors, marker='o', color='red', linewidth=2)
            ax.set_xlabel('Future Time Step')
            ax.set_ylabel('Prediction Error (normalized)')
            ax.set_title('Prediction Error Over Time')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=errors.mean(), color='orange', linestyle='--', 
                      label=f'Mean Error: {errors.mean():.4f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / f'sample_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization {i+1}/{num_samples}")


def compare_datasets(datasets_dict):
    """Compare motion characteristics between datasets."""
    print("\n" + "="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    # Analyze all datasets
    results = {}
    for name, dataset in datasets_dict.items():
        if dataset is not None:
            vel, acc = analyze_motion_characteristics(dataset, name, num_samples=500)
            results[name] = {'velocity': vel, 'acceleration': acc}
    
    if len(results) == 0:
        print("No datasets to compare!")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Define colors for different datasets
    colors = {'MOT17': 'blue', 'MOT20': 'green', 'DanceTrack': 'red'}
    
    # Velocity comparison
    ax = axes[0]
    for name, data in results.items():
        color = colors.get(name, 'gray')
        ax.hist(data['velocity'], bins=50, alpha=0.5, label=name, color=color, density=True)
    ax.set_xlabel('Velocity Magnitude (normalized)')
    ax.set_ylabel('Density')
    ax.set_title('Velocity Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Acceleration comparison
    ax = axes[1]
    for name, data in results.items():
        color = colors.get(name, 'gray')
        ax.hist(data['acceleration'], bins=50, alpha=0.5, label=name, color=color, density=True)
    ax.set_xlabel('Acceleration Magnitude (normalized)')
    ax.set_ylabel('Density')
    ax.set_title('Acceleration Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved dataset comparison plot to 'dataset_comparison.png'")
    
    # Statistical comparison
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISON")
    print(f"{'='*80}")
    
    # Compare all datasets pairwise
    dataset_names = list(results.keys())
    if len(dataset_names) >= 2:
        # Use first dataset as baseline
        baseline = dataset_names[0]
        baseline_vel = results[baseline]['velocity']
        baseline_acc = results[baseline]['acceleration']
        
        for name in dataset_names[1:]:
            print(f"\n{name} / {baseline} Ratios:")
            comp_vel = results[name]['velocity']
            comp_acc = results[name]['acceleration']
            
            print(f"  Velocity:")
            print(f"    Mean: {comp_vel.mean() / baseline_vel.mean():.2f}x")
            print(f"    Std: {comp_vel.std() / baseline_vel.std():.2f}x")
            print(f"    Max: {comp_vel.max() / baseline_vel.max():.2f}x")
            
            print(f"  Acceleration:")
            print(f"    Mean: {comp_acc.mean() / baseline_acc.mean():.2f}x")
            print(f"    Std: {comp_acc.std() / baseline_acc.std():.2f}x")
            print(f"    Max: {comp_acc.max() / baseline_acc.max():.2f}x")
            
            # Interpretation
            if comp_vel.mean() / baseline_vel.mean() > 2:
                print(f"  ⚠️  {name} has MUCH higher velocities than {baseline} - model needs more capacity!")
            if comp_acc.mean() / baseline_acc.mean() > 3:
                print(f"  ⚠️  {name} has MUCH higher accelerations than {baseline} - need motion features!")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    print("Loading datasets...")
    
    datasets = {}
    
    if args.mot17_path:
        mot17_dataset = GTSequenceDataset.from_roots(
            root_dirs=[args.mot17_path],
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            seq_total_len=args.seq_total_len,
            use_motion_features=args.use_motion_features
        )
        print(f"MOT17: {len(mot17_dataset)} samples")
        datasets['MOT17'] = mot17_dataset
    else:
        mot17_dataset = None
    
    if args.mot20_path:
        mot20_dataset = GTSequenceDataset.from_roots(
            root_dirs=[args.mot20_path],
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            seq_total_len=args.seq_total_len,
            use_motion_features=args.use_motion_features
        )
        print(f"MOT20: {len(mot20_dataset)} samples")
        datasets['MOT20'] = mot20_dataset
    else:
        mot20_dataset = None
    
    if args.dancetrack_path:
        dance_dataset = GTSequenceDataset.from_roots(
            root_dirs=[args.dancetrack_path],
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            seq_total_len=args.seq_total_len,
            use_motion_features=args.use_motion_features
        )
        print(f"DanceTrack: {len(dance_dataset)} samples")
        datasets['DanceTrack'] = dance_dataset
    else:
        dance_dataset = None
    
    # Compare datasets
    if len(datasets) >= 2:
        compare_datasets(datasets)
    
    # Load model and visualize predictions if checkpoint provided
    if args.checkpoint:
        print(f"\nLoading model from {args.checkpoint}...")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_args = checkpoint.get('args', {})
        
        # Create model
        input_dim = 13 if model_args.get('use_motion_features', True) else 5
        model = MotionTransformer(
            input_dim=input_dim,
            d_model=model_args.get('d_model', 256),
            nhead=model_args.get('nhead', 8),
            num_layers=model_args.get('num_layers', 6),
            dim_ff=model_args.get('dim_ff', 1024),
            dropout=model_args.get('dropout', 0.1),
            use_residual_prediction=model_args.get('use_residual', True)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
        
        # Visualize predictions for all datasets
        for dataset_name, dataset in datasets.items():
            if dataset is not None:
                print(f"\nVisualizing {dataset_name} predictions...")
                save_dir = f'visualizations/{dataset_name.lower()}'
                visualize_sample_predictions(
                    model, dataset, device,
                    num_samples=args.num_samples,
                    save_dir=save_dir
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and analyze motion predictions")
    
    parser.add_argument("--mot17_path", type=str, help="Path to MOT17 dataset")
    parser.add_argument("--mot20_path", type=str, help="Path to MOT20 dataset")
    parser.add_argument("--dancetrack_path", type=str, help="Path to DanceTrack dataset")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    
    parser.add_argument("--seq_in_len", type=int, default=20)
    parser.add_argument("--seq_out_len", type=int, default=10)
    parser.add_argument("--seq_total_len", type=int, default=30)
    parser.add_argument("--use_motion_features", action="store_true", default=True)
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of samples to visualize")
    
    args = parser.parse_args()
    main(args)

