#training
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import glob
import sys
from pathlib import Path
from scipy import stats
import os
from tqdm import tqdm
import torch.optim as optim
from typing import Dict, List
from models import  CNNController,TemporalCNNAttention, TransformerController, RNNControllerPatch, CausalTemporalAttention, ImprovedBurgersNet
# from evaluation import BurgersMetrics


# DT = 0.0025        # Time step size (dt) - common for Burgers'
# DX = 2 * np.pi / 128 # Spatial step size (dx) - common for Burgers' 
# # Since N = 128

# -------------------- Akash model's parameters: ------------------------------
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# --- Grid helpers: infer per-batch dx and use the generator's dt ---
def infer_dx_from_P(P: int, x_min: float = -5.0, x_max: float = 5.0):
    # Your generator uses x in [-5, 5]; adjust if you change the domain
    return (x_max - x_min) / (P - 1)

GEN_DT = 0.01  # This matches burgers_multipleIC.py (dt = t_max/(T-1) with t_max=1.0)

# --------------------------------------------------------------------------------

def extract_spatial_patches(field_batch, patch_radius: int = 1):
    """
    Simple CNN version:
    field_batch: (B, N)
    -> patches: (B, N, patch_size)
    """
    B, N = field_batch.shape
    x = field_batch.unsqueeze(1)  # (B, 1, N)
    pad = nn.ReplicationPad1d(patch_radius)
    padded = pad(x)               # (B, 1, N+2r)
    patches = padded.unfold(2, 2 * patch_radius + 1, 1)  # (B, 1, N, P)
    return patches.squeeze(1) # (B, N, P)


def build_patches_from_sequence(fields_seq, r: int, patch_size: int):
    """
    sam_cnn version:
    fields_seq: (B, L, N) -> (B*N, L, patch_size)
    """
    B, L, N = fields_seq.shape
    patches_list = []
    for l in range(L):
        field_l = fields_seq[:, l, :]                       # (B, N)
        padded_l = F.pad(field_l, (r, r), mode='replicate') # (B, N+2r)
        patches_l = padded_l.unfold(1, patch_size, 1)       # (B, N, P)
        patches_list.append(patches_l)
    patches_seq = torch.stack(patches_list, dim=2)          # (B, N, L, P)
    return patches_seq.reshape(B * N, L, patch_size)



# ---------- Training loop for RNN ----------

def train_rnn_patch(
    model: RNNControllerPatch,
    dataloader,
    device: torch.device,
    chunk_size: int = 3,
    num_epochs: int = 30,
    patch_radius: int = 1,
    resolution=None,
):
    """
    Loop inspired by rnn.py:
    - we take temporal chunks (chunk_size) as input
    - target: field at time t+chunk_size
    """
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    patch_size = 2 * patch_radius + 1

    epoch_losses: List[float] = []

    # --- path for saving loss plots (same convention as TCAN) ---
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "saved_results" / "new_dataset_results"
    os.makedirs(output_dir, exist_ok=True)
    res_tag = f"_{resolution}" if resolution is not None else ""

    for epoch in range(num_epochs):
        total_loss = 0.0
        for init_field, true_traj, nu in dataloader:
            true_traj = true_traj.to(device)  # (B, T, N)
            nu = nu.to(device)                # (B, 1)
            B, T, N = true_traj.shape

            all_preds = []
            all_targets = []
            for t in range(T - chunk_size):
                current_chunk = true_traj[:, t : t + chunk_size, :]
                next_true = true_traj[:, t + chunk_size, :]

                patches = build_patches_from_sequence(current_chunk, patch_radius, patch_size)
                nu_expanded = nu.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 1)
                pred_next = model(patches, nu_expanded).reshape(B, N)

                all_preds.append(pred_next)
                all_targets.append(next_true)

            if not all_preds:
                continue
            pred_traj   = torch.stack(all_preds,   dim=1)
            target_traj = torch.stack(all_targets, dim=1)

            loss = criterion(pred_traj, target_traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"[RNN] Epoch {epoch+1}/{num_epochs} - MSE: {avg_loss:.6e}")

        # --- loss plot every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(epoch_losses, 'b-o', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title(f'Transformer Training Loss (Epoch {epoch+1}){" - " + resolution if resolution else ""}')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plt.tight_layout()
            plt.savefig(output_dir / f"transformer_loss_epoch_{epoch+1}{res_tag}.png", dpi=100, bbox_inches='tight')
            plt.close()

    return model, epoch_losses



# ------------ Recurrent CNN training loop -----------------------------

# def train_cnn_controller(cnn_controller, train_loader, test_loader, num_epochs=30, roll_out_size=10, device='cuda', plot_learning_progress=None):
#     """
#     Complete training and evaluation loop for CNN controller on viscosity trajectories.
    
#     Args:
#         cnn_controller: PyTorch CNN model
#         train_loader: DataLoader with (initial_fields_batch, true_trajectories_batch, viscosities_batch)
#         test_loader: DataLoader for evaluation with (initial_field, true_trajectory, viscosity_val)
#         optimizer: PyTorch optimizer (e.g., torch.optim.Adam(cnn_controller.parameters()))
#         mse_loss: MSE loss function (nn.MSELoss())
#         num_epochs: Number of training epochs [default: 30]
#         roll_out_size: Number of autoregressive rollout steps [default: 10]
#         device: torch device ('cuda' or 'cpu') [default: 'cuda']
#         plot_learning_progress: Optional function for visualization (true_traj, pred_traj, epoch)
    
#     Returns:
#         epoch_losses: List of average losses per epoch

#     """
#     epoch_losses = []
#     mse_loss = nn.MSELoss() # LOSS FUNCTION
#     optimizer = torch.optim.Adam(cnn_controller.parameters(), lr=1e-4, weight_decay=1e-4) # OPTIMIZER
#     for epoch in range(num_epochs):
#         cnn_controller.train()
#         epoch_loss = 0.0

#         # Training: iterate over training data (viscosity trajectories)
#         for initial_fields_batch, true_trajectories_batch, viscosities_batch in train_loader:
#             initial_fields_batch = initial_fields_batch.to(device)  # (B, N)
#             true_trajectories_batch = true_trajectories_batch.to(device)  # (B, T, N)
            
#             _, T, N = true_trajectories_batch.shape
#             total_loss = 0.0
#             num_rollouts = 0
            
#             # Multiple rollouts per trajectory (every 10 steps)
#             for t in range(0, T - roll_out_size, 10):
#                 # Start from true field f_t
#                 current_field = true_trajectories_batch[:, t, :]  # (B, N)
                
#                 # Autoregressive rollout: f_{t+1}, ..., f_{t+roll_out_size}
#                 for roll_step in range(roll_out_size):
#                     next_field = cnn_controller(current_field)  # (B, N)
#                     current_field = next_field  # Recurrent update
                
#                 # Compare prediction at rollout endpoint
#                 prediction_at_rollout = current_field  # f_{t+roll_out_size}
#                 true_at_rollout = true_trajectories_batch[:, t + roll_out_size, :]
#                 rollout_loss = mse_loss(prediction_at_rollout, true_at_rollout)
#                 total_loss += rollout_loss
#                 num_rollouts += 1

#             # Average loss over rollouts for this batch, then backprop
#             loss = total_loss / num_rollouts
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(cnn_controller.parameters(), max_norm=1.0)
#             optimizer.step()
#             epoch_loss += loss.item()
        
#         # Epoch statistics
#         avg_loss = epoch_loss / len(train_loader)
#         epoch_losses.append(avg_loss)
#         print(f"Epoch {epoch+1}, Loss: {avg_loss}")

#         # Evaluation every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             evaluate_cnn_model(cnn_controller, test_loader, device, epoch, plot_learning_progress)
        
#         cnn_controller.train()  # Ensure training mode
    
#     return epoch_losses


# def evaluate_cnn_model(cnn_controller, test_loader, device, epoch, plot_learning_progress=None):
#     """Internal evaluation with 1-step MSE and full trajectory metrics."""
#     cnn_controller.eval()
#     with torch.no_grad():
#         for sample_idx, (initial_field, true_trajectory, viscosity_val) in enumerate(test_loader):
#             initial_field = initial_field.to(device)  # (1, N)
#             true_trajectory = true_trajectory.to(device)  # (1, T, N)
            
#             T_test, N_test = true_trajectory.shape[1], true_trajectory.shape[2]

#             # 1-step prediction MSE
#             f0_true = true_trajectory[:, 0, :]
#             f1_true = true_trajectory[:, 1, :]
#             f1_pred = cnn_controller(f0_true)
#             mse_1step = torch.mean((f1_true - f1_pred)**2).item()
#             print(f"[Epoch {epoch+1}] 1-step MSE: {mse_1step:.4e}")

#             # Full autoregressive trajectory prediction
#             pred_trajectory = []
#             current_field = initial_field
#             for t in range(T_test - 1):
#                 next_field = cnn_controller(current_field)
#                 pred_trajectory.append(next_field)
#                 current_field = next_field
            
#             # Stack and align shapes for comparison: (T, N)
#             pred_trajectory = torch.stack(pred_trajectory, dim=0).squeeze(1)  # (T-1, N)
#             init_2d = initial_field.squeeze(0).unsqueeze(0)  # (1, N)
#             pred_trajectory_2d = torch.cat([init_2d, pred_trajectory], dim=0)  # (T, N)
#             true_trajectory_2d = true_trajectory.squeeze(0)  # (T, N)

#             # Visualize first test sample only
#             if sample_idx == 0:
#                 print("="*50)
#                 print(f"Visualization of learning progress at epoch: {epoch + 1}")
                
#                 mse_val = torch.mean((true_trajectory_2d - pred_trajectory_2d)**2).item()
#                 print(f"[Epoch {epoch+1}] global MSE on test traj: {mse_val:.4e}")
                
#                 if plot_learning_progress is not None:
#                     plot_learning_progress(true_trajectory_2d, pred_trajectory_2d, epoch + 1)
                
#                 # Optional quality metrics (requires psnr/ssim functions)
#                 # psnr_val = psnr(true_trajectory_2d, pred_trajectory_2d, max_val=1.0)
#                 # ssim_val = ssim(true_trajectory_2d, pred_trajectory_2d, val_range=1.0)
#                 # print(f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
                
#                 print("="*50 + "\n")
    
#     cnn_controller.train()


def training_cnn(model, train_loader, optimizer, num_epochs, window_size, rollout_depth_max):
    device = next(model.parameters()).device
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    mse_loss = nn.MSELoss()
    
    print("\nTraining with Stability Enhancements...")
    pbar = tqdm(range(num_epochs))
    
    history = {'loss': [], 'energy_loss': []}
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_energy_loss = 0.0
        
        # Curriculum learning: gradually increase rollout depth
        rollout_depth = min(rollout_depth_max, max(4, int(rollout_depth_max * (epoch / num_epochs)**1.5)))
        
        for initial_fields_batch, true_trajectories_batch, _ in train_loader:
            true_trajectories_batch = true_trajectories_batch.to(device)
            
            B, T_traj, N_points = true_trajectories_batch.shape
            dx = infer_dx_from_P(N_points)
            total_loss = 0.0
            num_rollouts = 0
            
            max_start_t = T_traj - rollout_depth - window_size - 1
            if max_start_t <= 0: continue
                
            # Iterate over subsampled starting steps for rollouts
            for t_start in range(0, max_start_t, 10):
                
                current_window = true_trajectories_batch[:, t_start : t_start + window_size, :]
                
                # --- Noise Injection (Pushforward) ---
                if epoch > 5:
                    noise_scale = 0.01 * min(1.0, (epoch - 5) / 50)
                    current_window = current_window + torch.randn_like(current_window) * noise_scale
                
                # --- Rollout over 'rollout_depth' steps ---
                for roll_step in range(rollout_depth):
                    t_target = t_start + window_size + roll_step
                    
                    # 1. Prediction
                    pred = model(current_window) # (B, N)
                    target = true_trajectories_batch[:, t_target, :] # (B, N)
                    
                    # --- A. PHYSICS-INFORMED LOSS (Energy Dissipation) ---
                    pred_energy = compute_energy(pred, dx)
                    prev_energy = compute_energy(current_window[:, -1, :].detach(), dx)
                    
                    # Penalize energy increase (dissipation): $\max(0, E_{t+1} - E_t)$
                    energy_increase = torch.relu(pred_energy - prev_energy)
                    energy_penalty = torch.mean(energy_increase)
                    
                    # --- B. GRADIENT LOSS (Shocks) ---
                    pred_grad = spatial_gradient(pred, dx)
                    target_grad = spatial_gradient(target, dx)
                    grad_loss = torch.mean((pred_grad - target_grad)**2)
                    
                    # --- C. COMBINED LOSS ---
                    step_loss = mse_loss(pred, target) + 0.1 * grad_loss + 0.05 * energy_penalty
                    
                    total_loss += step_loss
                    epoch_energy_loss += energy_penalty.item()
                    
                    # 2. Recurrent Update (Window Shift)
                    teacher_forcing_ratio = max(0.0, 1.0 - epoch / 100) 
                    
                    if np.random.random() < teacher_forcing_ratio:
                        next_state = target.unsqueeze(1) # Teacher forcing
                    else:
                        next_state = pred.unsqueeze(1) # Self-prediction
                        
                    current_window = torch.cat([current_window[:, 1:, :], next_state], dim=1).detach()
                    num_rollouts += 1

            # Backpropagation
            if num_rollouts > 0:
                loss = total_loss / num_rollouts
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(train_loader) + 1e-6)
        avg_e_loss = epoch_energy_loss / (len(train_loader) * num_rollouts + 1e-6)
        history['loss'].append(avg_loss)
        pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.4e} | E_pen: {avg_e_loss:.4e} | Rollout: {rollout_depth}")
        
    return model, history


# ------------ 2D training -----------------------------

def build_patches_from_sequence_2d(seq, patch_radius: int):
    """
    seq: (B, L, C, H, W)
    return:
      patches: (B, L, H, W, P) with P = C * (2r+1)^2
    """
    B, L, C, H, W = seq.shape
    k = 2 * patch_radius + 1

    # on fusionne B et L pour unfold
    x = seq.reshape(B * L, C, H, W)                    # (B*L, C, H, W)
    x = F.pad(x, (patch_radius, patch_radius, patch_radius, patch_radius), mode="replicate")
    # unfold -> (B*L, C*k*k, H*W)
    patches = F.unfold(x, kernel_size=k, stride=1)     # (B*L, C*k*k, H*W)
    patches = patches.transpose(1, 2)                  # (B*L, H*W, P)
    patches = patches.view(B, L, H, W, C * k * k)      # (B, L, H, W, P)
    return patches

def train_transformer2d_patch(
    model,
    dataloader,
    device: torch.device,
    chunk_size: int = 3,
    num_epochs: int = 30,
    patch_radius: int = 1,
    n_points: int = 2048,   # nb de pixels échantillonnés par batch
):
    """
        Entraînement (full-field):
            input  : chunk temporel (B, chunk, C, H, W) -> patches -> pooling -> modèle
            target : champ à t+chunk (B, C, H, W)
        Loss: MSE sur le champ complet.
    """
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_losses: List[float] = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0

        for init_field, true_traj, nu in dataloader:
            true_traj = true_traj.to(device)   # (B, T, C, H, W)
            nu = nu.to(device)                 # (B, 1)
            B, T, C, H, W = true_traj.shape

            optimizer.zero_grad()

            # on cumule la loss sur tous les pas t -> t+chunk
            loss_accum = 0.0
            steps = 0

            for t in range(T - chunk_size):
                chunk = true_traj[:, t:t+chunk_size]         # (B, L, C, H, W)
                target = true_traj[:, t+chunk_size]          # (B, C, H, W)

                # patches par pixel et par temps: (B, L, H, W, P)
                patches = build_patches_from_sequence_2d(chunk, patch_radius)  # (B, L, H, W, P)

                # pooling spatial pour obtenir une séquence globale (B, L, P)
                patch_seq = patches.mean(dim=(2, 3))  # moyenne sur H,W

                pred_field = model(patch_seq, nu)     # attendu: (B, C, H, W)

                if pred_field.shape != target.shape:
                    raise RuntimeError(
                        f"Shape mismatch: pred {tuple(pred_field.shape)} vs target {tuple(target.shape)}. "
                        "Vérifie out_channels et le format des données."
                    )

                loss_step = criterion(pred_field, target)
                loss_accum = loss_accum + loss_step
                steps += 1

            loss = loss_accum / max(1, steps)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        print(f"[TF2D] Epoch {epoch+1}/{num_epochs} - MSE: {avg_loss:.6e}")

    return epoch_losses



# ==========================================
# 3. Improved Training with Stability Tricks
# ==========================================
def compute_energy(u, dx):
    if u.dim() == 3:
            u = u.squeeze(1)
    return 0.5 * torch.sum(u * u, dim=-1) * dx


def spatial_gradient(u, dx):
    if u.dim() == 2:
        u = u.unsqueeze(1)

    u_right = torch.roll(u, -1, dims=-1)
    u_left = torch.roll(u, 1, dims=-1)
    return (u_right - u_left) / (2 * dx)

def plot_trajectory_comparison(model, test_batch, history_len, epoch=None, resolution=None):
    """
    Plots full trajectory: True vs Pred vs Error
    """
    model.eval()
    device = next(model.parameters()).device

    # Path for saving results
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "saved_results" / "new_dataset_results"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        _, target_batch, nu_batch = test_batch
        target_batch = target_batch.to(device)
        nu_batch = nu_batch.to(device)

        B, T, P = target_batch.shape
        dx = infer_dx_from_P(P)
        dt = GEN_DT

        preds = []
        current_window = target_batch[:, :history_len].clone()  # (B, history_len, N)

        if current_window.dim() == 2:
            current_window = current_window.unsqueeze(1).repeat(1, history_len, 1)
        for t in range(T - history_len):
            pred = model(current_window, nu_batch)
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)
            preds.append(pred)
            current_window = torch.cat([current_window[:, 1:, :], pred], dim=1)

        preds = torch.cat(preds, dim=1)
        full_pred_traj = torch.cat([target_batch[:, :history_len], preds], dim=1)
        true = target_batch

        pred_np  = full_pred_traj[0].cpu().numpy()
        true_np  = true[0].cpu().numpy()
        nu_val   = float(nu_batch[0].item())
        error_np = np.abs(pred_np - true_np)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        im0 = axes[0].imshow(true_np.T, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title(f"Ground Truth for nu = {nu_val:.3g}")
        axes[0].set_xlabel('Time step t')
        axes[0].set_ylabel('Position x')
        fig.colorbar(im0, ax=axes[0], fraction=0.046).set_label('u(x,t)')

        im1 = axes[1].imshow(pred_np.T, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title(f'Prediction for nu = {nu_val:.3g}')
        axes[1].set_xlabel('Time step t')
        axes[1].set_ylabel('Position x')
        fig.colorbar(im1, ax=axes[1], fraction=0.046).set_label('u(x,t)')

        im2 = axes[2].imshow(error_np.T, aspect='auto', cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('Time step t')
        axes[2].set_ylabel('Position x')
        fig.colorbar(im2, ax=axes[2], fraction=0.046).set_label('Error magnitude')

        plt.tight_layout()
        if epoch is not None:
            # ── only change: resolution tag in filename ──
            res_tag = f"_{resolution}" if resolution is not None else ""
            plt.savefig(output_dir / f"trajectory_epoch_{epoch}{res_tag}_nu{nu_val:.3g}.png", dpi=150)
        plt.show()


# Function to plot loss per epoch
def plot_epoch_losses(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], 'b-', label='Total Loss', linewidth=2)
    plt.plot(history['energy_loss'], 'r--', label='Energy Penalty', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# ==========================================
# 4. Corrected Evaluation Metrics
# ==========================================
class BurgersMetrics:
    """Evaluation metrics with corrected mass handling"""
    
    def __init__(self, dx, dt, nu):
        self.dx = dx
        self.dt = dt
        self.nu = nu
    
    def mse(self, pred, gt):
        return np.mean((pred - gt)**2)
    
    def relative_l2(self, pred, gt):
        return np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-10)
    
    def relative_l2_per_timestep(self, pred, gt):
        num = np.sqrt(np.sum((pred - gt)**2, axis=-1))
        den = np.sqrt(np.sum(gt**2, axis=-1))
        return num / (den + 1e-10)
    
    def psnr(self, pred, gt):
        mse_val = self.mse(pred, gt)
        data_range = gt.max() - gt.min()
        if mse_val < 1e-10:
            return float('inf')
        return 20 * np.log10(data_range / np.sqrt(mse_val))
    
    def ssim_score(self, pred, gt):
        if not HAS_SKIMAGE:
            return 0.0
        data_range = gt.max() - gt.min()
        return ssim(gt, pred, data_range=data_range)
    
    def correlation_per_timestep(self, pred, gt):
        correlations = []
        for t in range(len(pred)):
            if np.std(pred[t]) < 1e-10 or np.std(gt[t]) < 1e-10:
                correlations.append(0.0)
            else:
                corr, _ = stats.pearsonr(pred[t].flatten(), gt[t].flatten())
                correlations.append(corr)
        return np.array(correlations)
    
    def time_to_threshold(self, values, threshold):
        crossed = np.where(values < threshold)[0]
        return crossed[0] if len(crossed) > 0 else len(values)
    
    def is_stable(self, trajectory, bounds=(-5, 5)):
        return np.all(np.abs(trajectory) < bounds[1])
    
    def has_nan_or_inf(self, trajectory):
        return np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory))
    
    # CORRECTED: Mass metric
    def mass(self, u):
        """Mass: M = ∫u dx"""
        return np.sum(u, axis=-1) * self.dx
    
    def mass_conservation_error(self, trajectory):
        """
        CORRECTED: Use absolute mass change when initial mass is near zero.
        For sin(x) initial condition, ∫sin(x)dx = 0 over [0, 2π].
        """
        mass_t = self.mass(trajectory)
        initial_mass = mass_t[0]
        
        # If initial mass is near zero, use absolute change
        if np.abs(initial_mass) < 0.1:
            return np.abs(mass_t - initial_mass)  # Absolute, not relative
        else:
            return np.abs(mass_t - initial_mass) / np.abs(initial_mass)
    
    def energy(self, u):
        """Energy: E = ½∫u² dx"""
        return 0.5 * np.sum(u**2, axis=-1) * self.dx
    
    def energy_dissipation_error(self, pred, gt):
        pred_energy = self.energy(pred)
        gt_energy = self.energy(gt)
        return np.abs(pred_energy - gt_energy) / (gt_energy + 1e-10)
    
    def check_energy_monotonicity(self, trajectory):
        energy_t = self.energy(trajectory)
        energy_diff = np.diff(energy_t)
        return np.mean(energy_diff <= 1e-6)  # Allow small numerical tolerance
    
    def pde_residual(self, u_curr, u_next):
        u_t = (u_next - u_curr) / self.dt
        u_left = np.roll(u_curr, 1, axis=-1)
        u_right = np.roll(u_curr, -1, axis=-1)
        u_x = (u_right - u_left) / (2 * self.dx)
        u_xx = (u_right - 2*u_curr + u_left) / self.dx**2
        residual = u_t + u_curr * u_x - self.nu * u_xx
        return residual
    
    def mean_pde_residual(self, trajectory):
        residuals = []
        for t in range(len(trajectory) - 1):
            res = self.pde_residual(trajectory[t], trajectory[t+1])
            residuals.append(np.mean(np.abs(res)))
        return np.mean(residuals)
    
    def energy_spectrum(self, u):
        fft_u = np.fft.fft(u, axis=-1)
        spectrum = np.abs(fft_u)**2
        n = spectrum.shape[-1]
        return spectrum[..., :n//2]
    
    def spectrum_error(self, pred, gt):
        pred_spec = self.energy_spectrum(pred)
        gt_spec = self.energy_spectrum(gt)
        # Use log ratio for better scaling
        log_pred = np.log10(pred_spec + 1e-10)
        log_gt = np.log10(gt_spec + 1e-10)
        return np.mean(np.abs(log_pred - log_gt))
    
    def gradient_error(self, pred, gt):
        pred_grad = np.gradient(pred, self.dx, axis=-1)
        gt_grad = np.gradient(gt, self.dx, axis=-1)
        return np.mean(np.abs(pred_grad - gt_grad), axis=-1)
    
    def max_gradient(self, u):
        u_grad = np.gradient(u, self.dx, axis=-1)
        return np.max(np.abs(u_grad), axis=-1)
    

def compute_metrics(model, test_batch, history_len):
    """Compute BurgersMetrics on the first batch of test data."""
    model.eval()
    device = next(model.parameters()).device
    
    mse_list, rel_l2_list, psnr_list, ssim_list, corr_list = [], [], [], [], []
    mass_err_list, energy_mono_list, pde_res_list, max_grad_pred_list, max_grad_err_list = [], [], [], [], []

    with torch.no_grad():
        _, target_batch, nu_batch = test_batch
        target_batch = target_batch.to(device)
        nu_batch = nu_batch.to(device)
        
        
        B, T, P = target_batch.shape
        dx = infer_dx_from_P(P)
        dt = GEN_DT

        # Initialize history window
        current_window = target_batch[:, :history_len].clone()

        # Rollout predictions
        preds = []
        # Ensure current_window has 3 dimensions: (B, history_len, N)
        if current_window.dim() == 2:
            current_window = current_window.unsqueeze(1).repeat(1, history_len, 1)
        for t in range(T - history_len):
            pred = model(current_window, nu_batch)
            if pred.dim() == 2:       # (B, N) -> (B, 1, N)
                pred = pred.unsqueeze(1)
            elif pred.shape[1] != 1:  # emergency check
                pred = pred[:, :1, :]
            preds.append(pred)
            current_window = torch.cat([current_window[:, 1:, :], pred], dim = 1)
        
        preds = torch.cat(preds, dim=1).cpu() # (B, T, N)
        
        preds_np = preds.numpy()
        targets_np = target_batch[:,history_len:].cpu().numpy() # (B, T, N)
        
        # Compute metrics for each sample in the batch
        for b in range(B):
            nu_val = nu_batch[b].item() if isinstance(nu_batch, torch.Tensor) else nu_batch[b]
            metrics_obj = BurgersMetrics(dx = dx, dt = dt, nu=nu_val)

            pred_b = np.squeeze(preds_np[b])
            true_b = np.squeeze(targets_np[b])
            assert pred_b.shape == true_b.shape, f"{pred_b.shape} vs {true_b.shape}"

            mse_list.append(metrics_obj.mse(pred_b, true_b))
            rel_l2_list.append(metrics_obj.relative_l2(pred_b, true_b))
            psnr_list.append(metrics_obj.psnr(pred_b, true_b))
            ssim_list.append(metrics_obj.ssim_score(pred_b, true_b))
            corr_list.append(np.mean(metrics_obj.correlation_per_timestep(pred_b, true_b)))

            mass_err_list.append(np.mean(metrics_obj.mass_conservation_error(pred_b)))
            energy_mono_list.append(metrics_obj.check_energy_monotonicity(pred_b))
            pde_res_list.append(metrics_obj.mean_pde_residual(pred_b))
            max_grad_pred_list.append(metrics_obj.max_gradient(pred_b))
            max_grad_err_list.append(metrics_obj.gradient_error(pred_b, true_b))

    # Average across all samples
    results = {
        "MSE": np.mean(mse_list),
        "Relative_L2": np.mean(rel_l2_list),
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "Correlation": np.mean(corr_list),
        "Mass_Error (Abs)": np.mean(mass_err_list),
        "Energy_Monotonicity (Frac)": np.mean(energy_mono_list),
        "PDE_Residual (Mean)": np.mean(pde_res_list),
        "Max_Grad_Pred (Max)": np.mean(max_grad_pred_list),
        "Max_Grad_Error (Mean)": np.mean(max_grad_err_list),
    }

    return results


# def train_model_TCAN(train_loader, test_loader, history_len=20, num_epochs=100, save_dir='training_plots_TCAN', resolution=None, model = None):
    
#     os.makedirs(save_dir, exist_ok=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if model is None:
#         model = ImprovedBurgersNet(window_size=history_len, corr_clip=0.1, use_viscosity=True).to(device)
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
#     test_batch = next(iter(test_loader))

#     print("="*60)
#     print("TRAINING WITH EXTERNAL DATASET")
#     print(f"History length: {history_len}")
#     print("="*60)

#     history = {'loss': [], 'energy_loss': []}

#     for epoch in tqdm(range(num_epochs), desc="Training"):
#         model.train()
#         total_loss = 0
#         energy_loss_total = 0
#         num_batches = 0

#         if epoch < num_epochs * 0.4:
#             rollout_depth = 8
#         elif epoch < num_epochs * 0.75:
#             rollout_depth = 16
#         else:
#             rollout_depth = 64

#         noise_scale = 0.01 * min(1.0, epoch / (num_epochs * 0.5))

#         for batch_idx, batch in enumerate(train_loader):
#             _, target_batch, nu_batch = batch
#             target_batch = target_batch.to(device)
#             nu_batch = nu_batch.to(device)

#             B, T_traj, P = target_batch.shape
#             dx = infer_dx_from_P(P)
#             dt = GEN_DT

#             optimizer.zero_grad()
#             batch_loss = 0.0
#             batch_energy_loss = 0.0

#             current_window = target_batch[:, :history_len].clone()

#             if epoch > num_epochs * 0.3 and noise_scale > 0:
#                 current_window = current_window + torch.randn_like(current_window) * noise_scale

#             for k in range(rollout_depth):
#                 target = target_batch[:, history_len + k].unsqueeze(1)
#                 pred = model(current_window, nu_batch)

#                 mse = torch.mean((pred - target)**2)

#                 pred_grad   = spatial_gradient(pred, dx)
#                 target_grad = spatial_gradient(target, dx)
#                 grad_loss   = torch.mean((pred_grad - target_grad)**2)

#                 u_x          = spatial_gradient(pred, dx).squeeze(1)
#                 u_xx         = spatial_gradient(u_x, dx)
#                 prev_state   = current_window[:, -1, :]
#                 u_t          = (pred.squeeze(1) - prev_state) / dt
#                 pde_residual = u_t + pred.squeeze(1) * u_x - nu_batch * u_xx
#                 pde_loss     = torch.mean(pde_residual ** 2)

#                 mass_pred  = torch.sum(pred, dim=-1) * dx
#                 mass_prev  = torch.sum(prev_state.unsqueeze(1), dim=-1) * dx
#                 mass_loss  = torch.mean((mass_pred - mass_prev) ** 2)

#                 pred_energy    = compute_energy(pred, dx)
#                 prev_energy    = compute_energy(prev_state, dx)
#                 energy_increase = torch.relu(pred_energy - prev_energy)
#                 energy_penalty  = torch.mean(energy_increase)

#                 step_loss   = mse + 0.3 * grad_loss + 0.1 * pde_loss + 0.05 * mass_loss + 0.3 * energy_penalty
#                 batch_loss += step_loss
#                 batch_energy_loss += energy_penalty.item()

#                 if k < rollout_depth - 1:
#                     if np.random.random() < 0.1 and epoch < 400:
#                         next_state = target
#                     else:
#                         next_state = pred
#                     current_window = torch.cat([current_window[:, 1:, :], next_state], dim=1)

#             batch_loss        = batch_loss / rollout_depth
#             batch_energy_loss = batch_energy_loss / rollout_depth
#             batch_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             total_loss        += batch_loss.item()
#             energy_loss_total += batch_energy_loss
#             num_batches       += 1

#         scheduler.step()
#         total_loss        = total_loss / num_batches
#         energy_loss_total = energy_loss_total / num_batches

#         if epoch % 10 == 0:
#             history['loss'].append(total_loss)
#             history['energy_loss'].append(energy_loss_total)

#             fig, axes = plt.subplots(1, 2, figsize=(14, 4))
#             epochs_so_far = [i * 10 for i in range(len(history['loss']))]

#             axes[0].plot(epochs_so_far, history['loss'], 'b-o', linewidth=2, markersize=4)
#             axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Total Loss')
#             axes[0].set_title(f'Training Loss (Epoch {epoch})')
#             axes[0].grid(True, alpha=0.3); axes[0].set_yscale('log')

#             axes[1].plot(epochs_so_far, history['energy_loss'], 'r-o', linewidth=2, markersize=4)
#             axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Energy Penalty')
#             axes[1].set_title('Energy Dissipation')
#             axes[1].grid(True, alpha=0.3); axes[1].set_yscale('log')

#             plt.tight_layout()
#             # ── only change: resolution tag in filename ──
#             res_tag = f"_{resolution}" if resolution is not None else ""
#             plt.savefig(f'{save_dir}/loss_epoch_{epoch}{res_tag}.png', dpi=100, bbox_inches='tight')
#             plt.close()

#             print("=" * 50)
#             tqdm.write(f"Epoch {epoch}: Loss={total_loss:.4f}, Energy_pen={energy_loss_total:.4f}, Rollout={rollout_depth}")
#             print("=" * 50)
#             plot_trajectory_comparison(model, test_batch, history_len, epoch=epoch, resolution=resolution)
#             metrics_results = compute_metrics(model, test_batch, history_len)
#             print("=" * 50)
#             print(f"Metrics at Epoch {epoch}")
#             print("=" * 50)
#             for key, value in metrics_results.items():
#                 print(f"  {key}: {value:.4f}")

#     return model, history, history_len

def train_model_TCAN(train_loader, test_loader, history_len=20, num_epochs=100, save_dir='training_plots_TCAN', resolution=None, model=None):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = ImprovedBurgersNet(window_size=history_len, corr_clip=0.1, use_viscosity=True).to(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    test_batch = next(iter(test_loader))

    print("="*60)
    print("TRAINING WITH EXTERNAL DATASET")
    print(f"History length: {history_len}")
    print("="*60)

    history = {'loss': []}  # ← removed 'energy_loss' (no longer tracked)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        num_batches = 0

        noise_scale = 0.01 * min(1.0, epoch / (num_epochs * 0.5))

        for batch_idx, batch in enumerate(train_loader):
            _, target_batch, nu_batch = batch
            target_batch = target_batch.to(device)
            nu_batch = nu_batch.to(device)

            B, T_traj, P = target_batch.shape
            dx = infer_dx_from_P(P)

            # ── CHANGE 2: rollout depth now matches evaluation in final phase ──
            eval_depth = T_traj - history_len
            if epoch < num_epochs * 0.4:
                rollout_depth = 8
            elif epoch < num_epochs * 0.75:
                rollout_depth = 16
            else:
                rollout_depth = eval_depth

            optimizer.zero_grad()
            batch_loss = 0.0

            current_window = target_batch[:, :history_len].clone()

            if epoch > num_epochs * 0.3 and noise_scale > 0:
                current_window = current_window + torch.randn_like(current_window) * noise_scale

            for k in range(rollout_depth):
                target = target_batch[:, history_len + k].unsqueeze(1)
                pred = model(current_window, nu_batch)

                # ── CHANGE 3: relative-L2 + gradient loss only ──────────────
                pred_sq   = pred.squeeze(1)    # (B, P) whether pred is (B,P) or (B,1,P)
                target_sq = target.squeeze(1)  # (B, P)
                diff  = pred_sq - target_sq
                denom = target_sq.norm(p=2, dim=-1).clamp(min=1e-8)
                rel_l2 = (diff.norm(p=2, dim=-1) / denom).mean()

                pred_grad   = spatial_gradient(pred_sq.unsqueeze(1), dx)
                target_grad = spatial_gradient(target_sq.unsqueeze(1), dx)
                grad_loss   = torch.mean((pred_grad - target_grad)**2)

                step_loss   = rel_l2 + 0.1 * grad_loss
                # ────────────────────────────────────────────────────────────
                batch_loss += step_loss

                if k < rollout_depth - 1:
                    if np.random.random() < 0.1 and epoch < 400:
                        next_state = target
                    else:
                        next_state = pred
                    current_window = torch.cat([current_window[:, 1:, :], next_state], dim=1)

            batch_loss = batch_loss / rollout_depth
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += batch_loss.item()
            num_batches += 1

        scheduler.step()
        total_loss = total_loss / num_batches

        if epoch % 10 == 0:
            history['loss'].append(total_loss)

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            epochs_so_far = [i * 10 for i in range(len(history['loss']))]
            ax.plot(epochs_so_far, history['loss'], 'b-o', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Rel-L2 + Grad Loss')
            ax.set_title(f'Training Loss (Epoch {epoch})')
            ax.grid(True, alpha=0.3); ax.set_yscale('log')

            plt.tight_layout()
            res_tag = f"_{resolution}" if resolution is not None else ""
            plt.savefig(f'{save_dir}/loss_epoch_{epoch}{res_tag}.png', dpi=100, bbox_inches='tight')
            plt.close()

            print("=" * 50)
            tqdm.write(f"Epoch {epoch}: Loss={total_loss:.4f}, Rollout={rollout_depth}")
            print("=" * 50)
            plot_trajectory_comparison(model, test_batch, history_len, epoch=epoch, resolution=resolution)
            metrics_results = compute_metrics(model, test_batch, history_len)
            print("=" * 50)
            print(f"Metrics at Epoch {epoch}")
            print("=" * 50)
            for key, value in metrics_results.items():
                print(f"  {key}: {value:.4f}")

    return model, history, history_len