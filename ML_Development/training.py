# training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import torch.optim as optim
from typing import Dict, List
from models import  CNNController,TemporalCNNAttention, TransformerController, RNNControllerPatch

DT = 0.0025        # Time step size (dt) - common for Burgers'
DX = 2 * np.pi / 128 # Spatial step size (dx) - common for Burgers' 
# Since N = 128

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


# ==========================================
# 2. Physics Helper Functions
# ==========================================

def compute_energy(field, dx=DX):
    """Compute Kinetic Energy for Burgers': E = 0.5 * ∫u² dx"""
    return 0.5 * torch.sum(field**2, dim=-1) * dx

def spatial_gradient(field, dx=DX):
    """Central difference spatial gradient: du/dx"""
    if field.dim() == 2:
        field = field.unsqueeze(1)
        
    field_right = torch.roll(field, -1, dims=-1)
    field_left = torch.roll(field, 1, dims=-1)
    return (field_right - field_left) / (2 * dx)

# ---------- Training loop for RNN ----------

def train_rnn_patch(
    model: RNNControllerPatch,
    dataloader,
    device: torch.device,
    chunk_size: int = 3,
    num_epochs: int = 30,
    patch_radius: int = 1,
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

    for epoch in range(num_epochs):
        total_loss = 0.0
        for init_field, true_traj, nu in dataloader:
            true_traj = true_traj.to(device)  # (B, T, N)
            nu = nu.to(device)                # (B, 1)
            B, T, N = true_traj.shape

            all_preds = []
            all_targets = []
            for t in range(T - chunk_size):
                current_chunk = true_traj[:, t : t + chunk_size, :]    # (B, chunk_size, N)
                next_true = true_traj[:, t + chunk_size, :]            # (B, N)

                # patches: (B, chunk_size, N, P) -> (B*N, chunk_size, P)
                patches = build_patches_from_sequence(current_chunk, patch_radius, patch_size)
                # patches: (B*N, chunk_size, P)
                nu_expanded = nu.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 1)  # (B*N,1)
                pred_next = model(patches, nu_expanded).reshape(B, N)              # (B,N)

                all_preds.append(pred_next)
                all_targets.append(next_true)

            if not all_preds:
                continue
            pred_traj = torch.stack(all_preds, dim=1)      # (B, T', N)
            target_traj = torch.stack(all_targets, dim=1)  # (B, T', N)

            loss = criterion(pred_traj, target_traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"[RNN] Epoch {epoch+1}/{num_epochs} - MSE: {avg_loss:.6e}")

    return epoch_losses



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
                    pred_energy = compute_energy(pred)
                    prev_energy = compute_energy(current_window[:, -1, :].detach())
                    
                    # Penalize energy increase (dissipation): $\max(0, E_{t+1} - E_t)$
                    energy_increase = torch.relu(pred_energy - prev_energy)
                    energy_penalty = torch.mean(energy_increase)
                    
                    # --- B. GRADIENT LOSS (Shocks) ---
                    pred_grad = spatial_gradient(pred)
                    target_grad = spatial_gradient(target)
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