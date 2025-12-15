"""
Improved Burgers Equation Neural Solver

Key fixes:
1. Corrected mass metric (use absolute change, not relative when mass ≈ 0)
2. Longer rollout training for stability
3. Noise injection for robustness
4. Physics-informed regularization (energy should dissipate)
5. Spectral normalization for stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import glob
import sys
from pathlib import Path
import os
files_dir = Path(__file__).parent.parent / 'ML_Development'
sys.path.insert(0, str(files_dir))
from data_loading import *

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

DX_CONST = 2 * np.pi / 128
DT_CONST = 1.0 / 256.0 # Asumiendo T_final = 1.0 y 256 pasos de tiempo
GRAD_LOSS_WEIGHT = 1.0



# ==========================================
# Model with Stability Improvements
# ==========================================
class CausalTemporalAttention(nn.Module):
    def __init__(self, window_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Sequential(
            nn.Conv1d(1, embed_dim, 3, padding=1),
            nn.GELU()
        )
        self.query_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.key_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.val_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.norm = nn.GroupNorm(4, embed_dim)

    def forward(self, u_history):
        B, W, P = u_history.shape
        flat_hist = u_history.view(B * W, 1, P)
        features = self.embedding(flat_hist).view(B, W, self.embed_dim, P)

        last_frame_feat = features[:, -1, :, :]
        Q = self.query_proj(last_frame_feat)

        flat_feat = features.view(B*W, self.embed_dim, P)
        K = self.key_proj(flat_feat).view(B, W, self.embed_dim, P)
        V = self.val_proj(flat_feat).view(B, W, self.embed_dim, P)

        scores = torch.einsum('bcp,bwcp->bwp', Q, K) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=1)

        context = torch.einsum('bwp,bwcp->bcp', attn_weights, V)
        out = self.out_proj(context)
        return self.norm(out + last_frame_feat)


class ImprovedBurgersNet(nn.Module):
    def __init__(self, window_size=4, corr_clip=0.1):
        super().__init__()
        self.corr_clip = corr_clip
        embed_dim = 32
        
        self.attn = CausalTemporalAttention(window_size, embed_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, 5, padding=2)
        )
        
        # Initialize last layer to output near-zero initially
        # This helps with stability at the start of training
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, u_history):
        u_last = u_history[:, -1:, :]
        context_features = self.attn(u_history)
        raw_correction = self.decoder(context_features)
        
        # Tanh clipping for bounded corrections
        correction = torch.tanh(raw_correction) * self.corr_clip
        
        u_next = u_last + correction
        return u_next


# ==========================================
# 3. Improved Training with Stability Tricks
# ==========================================
def compute_energy(u):
    """Compute energy: E = 0.5 * ∫u² dx"""
    return 0.5 * torch.sum(u**2, dim=-1) * DX_CONST

def spatial_gradient(u):
    """Central difference gradient"""
    u_right = torch.roll(u, -1, dims=-1)
    u_left = torch.roll(u, 1, dims=-1)
    return (u_right - u_left) / (2 * DX_CONST)

def plot_trajectory_comparison(model, test_batch, history_len, epoch=None):
    """
    Plots full trajectory: True vs Pred vs Error
    """
    model.eval()
    device = next(model.parameters()).device

    # Path for saving results
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "saved_results"
    os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist


    with torch.no_grad():
        _, target_batch, nu_batch = test_batch
        target_batch = target_batch.to(device)
        nu_batch = nu_batch.to(device)
        B, T, N = target_batch.shape

        preds = []
        current_window = target_batch[:, :history_len].clone() # (B, history_len, N)

        if current_window.dim() == 2:
            # Expand to (B, history_len, N)
            current_window = current_window.unsqueeze(1).repeat(1, history_len, 1)
        for t in range(T - history_len):
            # Ensure shape is (B, history_len, N)
            pred = model(current_window)
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)
            preds.append(pred)
            current_window = torch.cat([current_window[:, 1:, :], pred], dim=1)

        preds = torch.cat(preds, dim = 1)
        full_pred_traj = torch.cat([target_batch[:, :history_len],preds], dim=1) # Forma final: (B, T_FULL, N)
        true = target_batch # (B, T, N)

        # Take first sample
        pred_np = full_pred_traj[0].cpu().numpy()  # (T, N)
        true_np = true[0].cpu().numpy()   # (T, N)
        nu_val = float(nu_batch[0].item()) # (1)
        error_np = np.abs(pred_np - true_np)    # (T, N)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Ground Truth Plot
        im0 = axes[0].imshow(true_np.T, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title(f"Ground Truth for nu = {nu_val:.3g}")
        axes[0].set_xlabel('Time step t')
        axes[0].set_ylabel('Position x')
        cbar = fig.colorbar(im0, ax=axes[0], fraction=0.046)
        cbar.set_label('u(x,t)')

        # Prediction Plot
        im1 = axes[1].imshow(pred_np.T, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title(f'Prediction for nu = {nu_val:.3g}')
        axes[1].set_xlabel('Time step t')
        axes[1].set_ylabel('Position x')
        cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046)
        cbar.set_label('u(x,t)')

        # Error Plot
        im2 = axes[2].imshow(error_np.T, aspect='auto', cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('Time step t')
        axes[2].set_ylabel('Position x')
        cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046)
        cbar.set_label('Error magnitude')

        plt.tight_layout()
        if epoch is not None:
            plt.savefig(output_dir / f"trajectory_epoch_{epoch}_nu{nu_val:.3g}.png", dpi=150)
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



def train_model(train_loader, test_loader, history_len=20, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedBurgersNet(history_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # Fetch a single test batch upfront for fast evaluation ---
    test_batch = next(iter(test_loader))

    print("="*60)
    print("TRAINING WITH EXTERNAL DATASET")
    print(f"History length: {history_len}")
    print("="*60)
    
    history = {'loss': [], 'energy_loss': []}

    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        energy_loss_total = 0
        num_batches = 0
        
        # Curriculum rollout depth
        if epoch < 200:
            rollout_depth = 8
        elif epoch < 400:
            rollout_depth = 16
        else:
            rollout_depth = 64
        
        
        noise_scale = 0.01 * min(1.0, epoch / 400) # NOISE

        for batch_idx, batch in enumerate(train_loader):
            _, target_batch, _ = batch
            target_batch = target_batch.to(device)    # (B, 256, P)
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_energy_loss = 0.0

            # 1. INICIALIZACIÓN DE LA VENTANA DE HISTORIA (current_window)
            # Usamos los primeros 'W' pasos de la trayectoria GT como historia inicial.
            # current_window forma (B, W, P)
            current_window = target_batch[:, :history_len].clone()

            if epoch > 100:
                current_window = current_window + torch.randn_like(current_window) * noise_scale
            
            for k in range(rollout_depth):
                # Target REAL para el paso T_predicho = WINDOW_SIZE + k
                # Forma del target: (B, P). Usamos unsqueeze(1) -> (B, 1, P)
                target = target_batch[:, history_len + k].unsqueeze(1)
                
                pred = model(current_window) # (B, 1, P)

                # MSE
                mse = mse = torch.mean((pred - target)**2)
                
                # Gradient matching
                pred_grad = spatial_gradient(pred)
                target_grad = spatial_gradient(target)
                grad_loss = torch.mean((pred_grad - target_grad)**2)
                
                prev_state = current_window[:, -1, :].clone() # (B, P)
                # Energy dissipation (E(t+1) <= E(t))
                pred_energy = compute_energy(pred.squeeze(1))
                prev_energy = compute_energy(prev_state)

                energy_increase = torch.relu(pred_energy - prev_energy)
                energy_penalty = torch.mean(energy_increase)
                
                # Combined loss
                step_loss =  mse + 0.1 * grad_loss + 0.5 * energy_penalty
                batch_loss += step_loss
                batch_energy_loss += energy_penalty.item() 
                
                # Update window for next step (with detach to limit backprop depth)
                if k < rollout_depth - 1:
                    # Occasionally use teacher forcing for stability
                    if np.random.random() < 0.1 and epoch < 400:
                        next_state = target  # Teacher forcing
                    else:
                        next_state = pred
                    current_window = torch.cat([current_window[:, 1:, :], next_state], dim=1)
            # End of rollout loop
            # Avg loss over rollout for current batch
            batch_loss = batch_loss / rollout_depth
            batch_energy_loss = batch_energy_loss / rollout_depth
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Add loss to epoch total loss:
            total_loss += batch_loss.item()
            energy_loss_total += batch_energy_loss
            num_batches += 1
            # end of batches

        # end of epoch
        total_loss = total_loss / num_batches
        energy_loss_total = energy_loss_total / num_batches
        if epoch % 10 == 0:
            history['loss'].append(total_loss)
            history['energy_loss'].append(energy_loss_total)
            print("=" * 50)
            tqdm.write(f"Epoch {epoch}: Loss={total_loss:.4f}, Energy_pen={energy_loss_total:.4f}, Rollout={rollout_depth}")
            print("=" * 50)
            plot_trajectory_comparison(model, test_batch, history_len, epoch = epoch)
            # Compute metrics
            metrics_results = compute_metrics(model, test_batch, history_len)
            print("=" * 50)
            print(f"Metrics at Epoch {epoch}")
            print("=" * 50)
            for key, value in metrics_results.items():
                print(f"  {key}: {value:.4f}")
        
    return model, history, history_len


# ==========================================
# 4. Evaluation (simplified for external data)
# ==========================================
def evaluate_model(model, test_loader, history_len):
    model.eval()
    device = next(model.parameters()).device
    
    all_mse = []
    
    with torch.no_grad():
        for _, target_batch, _ in test_loader:
            target_batch = target_batch.to(device)
            B, T, N = target_batch.shape

            preds = []
            current_window = target_batch[:, :history_len].clone()
            
            # Autoregressive rollout
            for _ in range(T - history_len):  # Fixed evaluation horizon
                pred = model(current_window)
                if pred.dim() == 2:
                    pred = pred.unsqueeze(1)   # emergency guard
                preds.append(pred)
                current_window = torch.cat([current_window[:, 1:, :], pred], dim=1)
                if len(preds) >= target_batch.shape[1]:
                    break
            
            preds = torch.cat(preds, dim = 1)
            target = target_batch[:,history_len:]
            mse = torch.nn.functional.mse_loss(preds, target).item()
            all_mse.append(mse)
    
    print(f"Test MSE: {np.mean(all_mse):.4e} ± {np.std(all_mse):.4e}")
    return np.mean(all_mse)

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
        
        B, T, N = target_batch.shape
        # Initialize history window
        current_window = target_batch[:, :history_len].clone()

        # Rollout predictions
        preds = []
        # Ensure current_window has 3 dimensions: (B, history_len, N)
        if current_window.dim() == 2:
            current_window = current_window.unsqueeze(1).repeat(1, history_len, 1)
        for t in range(T - history_len):
            pred = model(current_window)
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
            metrics_obj = BurgersMetrics(dx = DX_CONST, dt = DT_CONST, nu=nu_val)

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



if __name__ == "__main__":
    # Your data loading
    project_root = Path(__file__).parent.parent
    train_data_dir = project_root / "saved_dataset" / "generated_1d_burgers" / "train"
    test_data_dir = project_root / "saved_dataset" / "generated_1d_burgers" / "test"
    
    history_len = 20
    batch_size = 32
    
    train_loader, test_loader = create_generated_dataloaders_from_folders(
        train_dir=str(train_data_dir),
        test_dir=str(test_data_dir),
        history_len=history_len,
        batch_size=batch_size
    )

    # Train
    model, history, history_len = train_model(train_loader, test_loader, history_len)
    
    # Fetch a single test batch upfront for fast evaluation ---
    test_batch = next(iter(test_loader))
    # Evaluate
    test_mse = evaluate_model(model, test_loader, history_len)
    plot_trajectory_comparison(model, test_batch, history_len, epoch = 100)
    print(f"Final Test MSE: {test_mse:.4e}")
    plot_epoch_losses(history, project_root/"saved_results/losses_over_epochs.png")
    # Final evaluation on test set
    final_metrics = compute_metrics(model, test_batch, history_len)
    print("=" * 50)
    print("Final Test Metrics:")
    print("=" * 50)
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")