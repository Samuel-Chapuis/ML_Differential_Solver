# evaluation.py
import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from training import extract_spatial_patches, BurgersMetrics
import numpy as np
import math
from scipy import stats
from visualization import plot_learning_progress, show_field



DT = 0.0025        # Time step size (dt) - common for Burgers'
DX = 2 * np.pi / 128 # Spatial step size (dx) - common for Burgers' 
# Since N = 128

# -------------------- Akash model's parameters: ------------------------------
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

DX_CONST = 2 * np.pi / 128
DT_CONST = 1.0 / 256.0 # Asumiendo T_final = 1.0 y 256 pasos de tiempo
GRAD_LOSS_WEIGHT = 1.0

# --------------------------------------------------------------------------------


import torch
import matplotlib.pyplot as plt


# ============================================================
# MAE local per time -> shape (T,)
# ============================================================
def local_mae_per_time(true_traj: torch.Tensor, pred_traj: torch.Tensor) -> torch.Tensor:
    """
    true_traj, pred_traj: (T, N)
    Retour: (T,) = MAE(t) moyennée sur N
    """
    return torch.mean(torch.abs(true_traj - pred_traj), dim=1)


# ============================================================
# Relative L2 error per time -> shape (T,)
# ============================================================
def relative_l2_error_per_time(
    true_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    true_traj, pred_traj: (T, N)
    Retour: (T,) = ||pred-true||2 / (||true||2 + eps), calculé par temps t
    """
    num = torch.linalg.norm(pred_traj - true_traj, ord=2, dim=1)          # (T,)
    den = torch.linalg.norm(true_traj, ord=2, dim=1).clamp_min(eps)       # (T,)
    return num / den


# ============================================================
# 2D metrics per time -> shape (T,)
# ============================================================
def local_mae_per_time_2d(true_traj: torch.Tensor, pred_traj: torch.Tensor) -> torch.Tensor:
    """
    true_traj, pred_traj: (T, C, H, W)
    Retour: (T,) = MAE(t) moyennée sur (C, H, W)
    """
    return torch.mean(torch.abs(true_traj - pred_traj), dim=(1, 2, 3))


def relative_l2_error_per_time_2d(
    true_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    true_traj, pred_traj: (T, C, H, W)
    Retour: (T,) = ||pred-true||2 / (||true||2 + eps), calculé par temps t
    """
    T = true_traj.shape[0]
    diff = (pred_traj - true_traj).reshape(T, -1)
    ref = true_traj.reshape(T, -1)
    num = torch.linalg.norm(diff, ord=2, dim=1)
    den = torch.linalg.norm(ref, ord=2, dim=1).clamp_min(eps)
    return num / den


# ============================================================
#  Error evolution on loaderx
# ============================================================
def error_evolution_on_loader(
    model,
    data_loader,
    device,
    patch_radius: int,
    chunk_size: int = 3,
    show_plot: bool = True,
    metric: str = "mae",   # "mae" ou "rel_l2"
):
    """
    Calcule l'évolution de l'erreur moyenne dans le temps, moyennée sur tout le data_loader.

    metric:
      - "mae"    : moyenne de |pred-true| sur l'espace
      - "rel_l2" : erreur relative L2 = ||pred-true||2 / ||true||2 par temps

    Retour:
      mean_error_per_time : numpy array (T,)
    """
    model.eval()

    sum_error_per_time = None
    n_samples = 0
    T_ref = None

    for init_field, traj_batch, nu_batch in data_loader:
        B, T, N = traj_batch.shape
        if T_ref is None:
            T_ref = T

        for b in range(B):
            true_traj = traj_batch[b].to(device)  # (T, N)
            nu_value = float(nu_batch[b].item())

            # Trajectoire prédite
            pred_traj = rollout_trajectory(
                model=model,
                true_traj=true_traj,
                nu_value=nu_value,
                device=device,
                patch_radius=patch_radius,
                chunk_size=chunk_size,
                verbose=False,
            )

            # Sécurité sur la longueur
            T_eff = min(true_traj.shape[0], pred_traj.shape[0], T_ref)
            true_traj_eff = true_traj[:T_eff, :]
            pred_traj_eff = pred_traj[:T_eff, :]

            # ---- Calcul erreur par temps (T_eff,)
            if metric.lower() == "mae":
                error_per_time = local_mae_per_time(true_traj_eff, pred_traj_eff)
                ylabel = "Mean absolute error"
                title_metric = "MAE"
            elif metric.lower() in ["rel_l2", "relative_l2", "l2_rel", "relative"]:
                error_per_time = relative_l2_error_per_time(true_traj_eff, pred_traj_eff)
                ylabel = "Relative L2 error"
                title_metric = "Relative L2"
            else:
                raise ValueError(f"Unknown metric='{metric}'. Use 'mae' or 'rel_l2'.")

            if sum_error_per_time is None:
                sum_error_per_time = torch.zeros(T_ref, device=device)

            sum_error_per_time[:T_eff] += error_per_time
            n_samples += 1

    if n_samples == 0:
        raise RuntimeError("error_evolution_on_loader: no samples in data_loader")

    mean_error_per_time = (sum_error_per_time / n_samples).cpu().numpy()

    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(mean_error_per_time, 'r-', linewidth=2)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.title(f"Error evolution over time ({title_metric}, averaged over test set)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return mean_error_per_time


# ============================================================
#  Error evolution on loader (2D)
# ============================================================
def error_evolution_on_loader_2d(
    model,
    data_loader,
    device,
    patch_radius: int,
    chunk_size: int = 3,
    show_plot: bool = True,
    metric: str = "mae",   # "mae" ou "rel_l2"
):
    """
    Calcule l'évolution de l'erreur moyenne dans le temps (2D), moyennée sur tout le data_loader.

    metric:
      - "mae"    : moyenne de |pred-true| sur (C, H, W)
      - "rel_l2" : erreur relative L2 = ||pred-true||2 / ||true||2 par temps

    Retour:
      mean_error_per_time : numpy array (T,)
    """
    model.eval()

    sum_error_per_time = None
    n_samples = 0
    T_ref = None

    for init_field, traj_batch, nu_batch in data_loader:
        B, T, C, H, W = traj_batch.shape
        if T_ref is None:
            T_ref = T

        for b in range(B):
            true_traj = traj_batch[b].to(device)  # (T, C, H, W)
            try:
                nu_value = float(nu_batch[b].item())
            except Exception:
                nu_value = float(nu_batch)

            pred_traj = rollout_trajectory_2d(
                model=model,
                true_traj=true_traj,
                nu_value=nu_value,
                device=device,
                patch_radius=patch_radius,
                chunk_size=chunk_size,
                verbose=False,
            )

            T_eff = min(true_traj.shape[0], pred_traj.shape[0], T_ref)
            true_traj_eff = true_traj[:T_eff]
            pred_traj_eff = pred_traj[:T_eff]

            if metric.lower() == "mae":
                error_per_time = local_mae_per_time_2d(true_traj_eff, pred_traj_eff)
                ylabel = "Mean absolute error"
                title_metric = "MAE"
            elif metric.lower() in ["rel_l2", "relative_l2", "l2_rel", "relative"]:
                error_per_time = relative_l2_error_per_time_2d(true_traj_eff, pred_traj_eff)
                ylabel = "Relative L2 error"
                title_metric = "Relative L2"
            else:
                raise ValueError(f"Unknown metric='{metric}'. Use 'mae' or 'rel_l2'.")

            if sum_error_per_time is None:
                sum_error_per_time = torch.zeros(T_ref, device=device)

            sum_error_per_time[:T_eff] += error_per_time
            n_samples += 1

    if n_samples == 0:
        raise RuntimeError("error_evolution_on_loader_2d: no samples in data_loader")

    mean_error_per_time = (sum_error_per_time / n_samples).cpu().numpy()

    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(mean_error_per_time, 'r-', linewidth=2)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.title(f"Error evolution over time ({title_metric}, averaged over test set)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return mean_error_per_time







def psnr(true: torch.Tensor, pred: torch.Tensor, max_val: float = 1.0) -> float:
    """
    true, pred : mêmes shapes (par ex. (T, N) ou (B, T, N))
    """
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val**2 / mse).item()


def _gaussian(window_size, sigma):
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel=1):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(true: torch.Tensor, pred: torch.Tensor, window_size=11, val_range=1.0) -> float:
    """
    SSIM 2D (pour cartes (space,time)).
    true, pred : (H, W) ou (B,1,H,W)
    """
    if true.dim() == 2:
        true = true.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)

    channel = true.size(1)
    window = _create_window(window_size, channel).to(true.device)

    mu1 = F.conv2d(true, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(true * true, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(true * pred, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    R² global.
    """
    ssr = torch.sum((target - pred) ** 2)
    mean_target = torch.mean(target)
    sst = torch.sum((target - mean_target) ** 2) + eps
    r2 = 1 - ssr / sst
    return r2.item()


def get_loader_nu_values(loader):
    """Return a sorted list of unique nu values found in a DataLoader or dataset.

    The loader can be a DataLoader or any iterable returning (init, traj, nu).
    """
    nus = set()
    for batch in loader:
        try:
            nu = batch[2]
        except Exception:
            _, _, nu = batch

        if isinstance(nu, torch.Tensor):
            for v in nu.reshape(-1).cpu().numpy():
                try:
                    nus.add(float(v))
                except Exception:
                    continue
        else:
            try:
                nus.add(float(nu))
            except Exception:
                continue
    return sorted(nus)


def resolve_nu_target(
    loader,
    plot_nu=None,
    selected_nu=None,
    list_nus: bool = False,
    verbose: bool = True,
):
    """Utility to resolve a nu target from loader and a plot specifier.

    Arguments:
      loader: data loader or iterable used to enumerate available nu values
      plot_nu: one of None | float | 'min'|'max'|'idx:N'|'random'|'list'|str->float
      selected_nu: if provided, priority target (returned directly)
      list_nus: if True -> print the list of nus and return None
      verbose: prints helpful messages

    Returns:
      float | None
    """
    nus = []
    try:
        nus = get_loader_nu_values(loader)
    except Exception:
        if verbose:
            print("resolve_nu_target: unable to enumerate nus from loader")

    if list_nus and verbose:
        print(f"Loader contains {len(nus)} unique nu values: {nus}")

    if selected_nu is not None:
        if verbose:
            print(f"Using provided selected_nu: {selected_nu}")
        return float(selected_nu)

    if plot_nu is None:
        return None

    # numeric types: choose closest if available
    if isinstance(plot_nu, (int, float)):
        if nus:
            val = float(plot_nu)
            closest = min(nus, key=lambda x: abs(x - val))
            if abs(closest - val) > 1e-8 and verbose:
                print(f"Requested nu {val} not in loader; using nearest available {closest}.")
            return float(closest)
        else:
            return float(plot_nu)

    if isinstance(plot_nu, str):
        s = plot_nu.strip().lower()
        if s == 'list':
            # already printed nus above
            return None
        if s == 'min' and nus:
            return float(min(nus))
        if s == 'max' and nus:
            return float(max(nus))
        if s.startswith('idx:') and nus:
            try:
                idx = int(s.split(':', 1)[1])
                if 0 <= idx < len(nus):
                    return float(nus[idx])
                else:
                    if verbose:
                        print(f"resolve_nu_target: index {idx} out of range (0..{len(nus)-1})")
                    return None
            except Exception:
                if verbose:
                    print("resolve_nu_target: invalid idx format; use 'idx:N'")
                return None
        if s == 'random' and nus:
            import random
            return float(random.choice(nus))
        # fallback: try parse numeric string
        try:
            val = float(s)
            if nus:
                closest = min(nus, key=lambda x: abs(x - val))
                if abs(closest - val) > 1e-8 and verbose:
                    print(f"Requested nu {val} not in loader; using nearest available {closest}.")
                return float(closest)
            else:
                return float(val)
        except Exception:
            if verbose:
                print(f"resolve_nu_target: could not parse PLOT_NU string '{plot_nu}'")
            return None


# def generate_model_predictions(model, train_loader, device, patch_radius,
#                                verbose: bool = True, chunk_size: int = 3):
#     """
#     Génère une trajectoire prédite à partir d'un batch du train_loader.

#     Compatible avec :
#       - CNNControllerPatch (mode "1 time step")
#       - RNNControllerPatch (mode séquentiel)
#       - TransformerController (mode séquentiel avec contexte)
#       - CNNSpaceTimeController (mode séquentiel spatio-temporel)
#     """
#     from training import build_patches_from_sequence, extract_spatial_patches

#     # --- On récupère un batch et on ne garde que le premier exemple ---
#     init_field, true_traj, nu = next(iter(train_loader))
#     true_traj = true_traj[0].to(device)          # (T, N)
#     nu_value = float(nu[0].item())
#     T, N = true_traj.shape

#     if verbose:
#         print(f"Generating predictions for nu = {nu_value:.4f}")
#         print(f"True trajectory shape: {true_traj.shape}")
#         print(f"Model type: {type(model).__name__}")

#     model.eval()
#     patch_size = 2 * patch_radius + 1

#     # --- Détection par nom de classe, pas par isinstance ---
#     model_name = type(model).__name__

#     # CNN 1-step : patch spatial (B, P) + nu
#     is_cnn_patch = model_name in ["CNNControllerPatch", "CNNController"]

#     # Tout le reste = modèles séquentiels (RNN, CNNHistory, CNNSpaceTime, Transformer…)
#     is_seq_model = not is_cnn_patch

#     preds = []

#     with torch.no_grad():

#         # ------------------------------------------------------
#         # 1) MODE SÉQUENTIEL : RNN / CNNHistory / SpaceTimeCNN
#         # ------------------------------------------------------
#         if is_seq_model:
#             if T <= chunk_size:
#                 if verbose:
#                     print(f"Warning: T={T} <= chunk_size={chunk_size}, "
#                           "on renvoie la vérité terrain comme prédiction.")
#                 pred_traj = true_traj.clone()
#             else:
#                 if verbose:
#                     print(f"Sequential mode: {T - chunk_size} prédictions "
#                           f"avec chunk_size={chunk_size}")

#                 for t in range(T - chunk_size):
#                     if verbose and t % 10 == 0:
#                         print(f"  step {t}/{T - chunk_size}")

#                     # Historique temporel : (1, chunk_size, N)
#                     current_chunk = true_traj[t:t + chunk_size, :].unsqueeze(0)

#                     # Patches spatio-temporels : (N, chunk_size, patch_size)
#                     patches = build_patches_from_sequence(
#                         current_chunk, patch_radius, patch_size
#                     )

#                     # Nu pour chaque patch spatial
#                     nu_vals = torch.full(
#                         (patches.size(0), 1),
#                         nu_value,
#                         device=device
#                     )  # (N, 1)

#                     # Prédiction du pas de temps suivant pour chaque point spatial
#                     pred_next = model(patches, nu_vals)  # (N,)
#                     preds.append(pred_next)

#                 # (T - chunk_size, N)
#                 pred_future = torch.stack(preds, dim=0)

#                 # On recolle les chunk_size premiers instants en vérité terrain
#                 pred_traj = torch.cat(
#                     [true_traj[:chunk_size, :], pred_future],
#                     dim=0
#                 )  # (T, N)

#         # ------------------------------------------------------
#         # 2) MODE CNN 1-STEP : CNNControllerPatch
#         # ------------------------------------------------------
#         elif is_cnn_patch:
#             if verbose:
#                 print(f"CNN mode: génération de {T - 1} pas de temps...")

#             for t in range(T - 1):
#                 if verbose and t % 10 == 0:
#                     print(f"  step {t}/{T - 1}")

#                 field_t = true_traj[t].unsqueeze(0)              # (1, N)
#                 patches = extract_spatial_patches(field_t, patch_radius)  # (1, N, P)
#                 patches_flat = patches.reshape(N, -1)            # (N, P)

#                 nu_vals = torch.full((N, 1), nu_value, device=device)
#                 pred_next = model(patches_flat, nu_vals)         # (N,)
#                 preds.append(pred_next)

#             pred_future = torch.stack(preds, dim=0)              # (T-1, N)
#             pred_traj = torch.cat([true_traj[0:1, :], pred_future], dim=0)

#         # ------------------------------------------------------
#         # 3) Autre type de modèle (par sécurité)
#         # ------------------------------------------------------
#         else:
#             raise TypeError(
#                 f"generate_model_predictions ne sait pas gérer le type de modèle {model_name} "
#                 f"(attendus : CNNControllerPatch, RNNControllerPatch, "
#                 f"TransformerController, CNNSpaceTimeController)."
#             )

#     if verbose:
#         print(f"\nGenerated predictions shape: {pred_traj.shape}")

#     return true_traj, pred_traj, nu_value

def evaluate_model_on_sample(model, train_loader, device, patch_radius, max_val=1.0, val_range=1.0, chunk_size=3, nu_target=None):
    """
    Evaluates a model on a sample and returns performance metrics.
    Compatible with CNN and RNN.
    
    Args:
        model: The trained CNN or RNN model
        train_loader: DataLoader containing the data
        device: PyTorch device (cuda or cpu) 
        patch_radius: Radius of spatial patches
        max_val: Maximum value for PSNR calculation
        val_range: Value range for SSIM calculation
            chunk_size: Size of temporal chunks for RNN (ignored for CNN)
            nu_target: If provided, tries to pick a sample with this viscosity value from the loader.
    
    Returns:
        dict: Dictionary containing metrics and trajectories
            - 'true_traj': Real trajectory
            - 'pred_traj': Predicted trajectory
            - 'nu_value': Viscosity value
            - 'psnr': PSNR score
            - 'ssim': SSIM score
            - 'mse': Mean squared error
            - 'r2': R² score
    """
    true_traj, pred_traj, nu_value = generate_model_predictions(
        model,
        train_loader,
        device,
        patch_radius,
        verbose=False,
        chunk_size=chunk_size,
        nu_target=nu_target,
    )
    
    # Calculate metrics
    psnr_score = psnr(true_traj, pred_traj, max_val=max_val)
    ssim_score = ssim(true_traj, pred_traj, val_range=val_range)
    mse_score = torch.mean((true_traj - pred_traj)**2).item()
    r2_score_val = r2_score(pred_traj, true_traj)
    
    return {
        'true_traj': true_traj,
        'pred_traj': pred_traj,
        'nu_value': nu_value,
        'psnr': psnr_score,
        'ssim': ssim_score,
        'mse': mse_score,
        'r2': r2_score_val
    }


def _ssim_2d_trajectory(true_traj: torch.Tensor, pred_traj: torch.Tensor, val_range: float = 1.0) -> float:
    """
    Compute SSIM averaged over time and channels for (T, C, H, W).
    """
    if true_traj.dim() != 4 or pred_traj.dim() != 4:
        raise ValueError("_ssim_2d_trajectory expects (T, C, H, W) tensors.")
    T, C, _, _ = true_traj.shape
    scores = []
    for t in range(T):
        for c in range(C):
            scores.append(ssim(true_traj[t, c], pred_traj[t, c], val_range=val_range))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_model_on_sample_2d(
    model,
    train_loader,
    device,
    patch_radius,
    max_val=1.0,
    val_range=1.0,
    chunk_size=3,
    nu_target=None,
):
    """
    2D evaluation on a sample.

    Returns dict with:
      - true_traj: (T, C, H, W)
      - pred_traj: (T, C, H, W)
      - nu_value
      - psnr, ssim, mse, r2
    """
    true_traj, pred_traj, nu_value = generate_model_predictions_2d(
        model,
        train_loader,
        device,
        patch_radius,
        verbose=False,
        chunk_size=chunk_size,
        nu_target=nu_target,
    )

    psnr_score = psnr(true_traj, pred_traj, max_val=max_val)
    ssim_score = _ssim_2d_trajectory(true_traj, pred_traj, val_range=val_range)
    mse_score = torch.mean((true_traj - pred_traj) ** 2).item()
    r2_score_val = r2_score(pred_traj, true_traj)

    return {
        'true_traj': true_traj,
        'pred_traj': pred_traj,
        'nu_value': nu_value,
        'psnr': psnr_score,
        'ssim': ssim_score,
        'mse': mse_score,
        'r2': r2_score_val
    }


def display_evaluation_results(evaluation_results, show_plots=True):
    """
    Displays evaluation metrics and error visualizations.
    
    Args:
        evaluation_results: Dictionary returned by evaluate_model_on_sample()
        show_plots: If True, displays error plots
        
    Returns:
        tuple: (true_traj, pred_traj) for later use if needed
    """
    true_traj = evaluation_results['true_traj']
    pred_traj = evaluation_results['pred_traj']
    
    print(f"Evaluation metrics:")
    print(f"  - PSNR: {evaluation_results['psnr']:.3f} dB")
    print(f"  - SSIM: {evaluation_results['ssim']:.3f}")
    print(f"  - MSE: {evaluation_results['mse']:.6f}")
    print(f"  - R²: {evaluation_results['r2']:.4f}")
    
    if show_plots:
        # Error visualization
        error = torch.abs(true_traj - pred_traj)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(error.cpu().numpy(), aspect='auto', cmap='hot')
        plt.colorbar(label='Absolute error')
        plt.xlabel('Spatial position')
        plt.ylabel('Time')
        plt.title('Absolute error map')
        
        plt.subplot(1, 2, 2)
        mean_error_per_time = torch.mean(error, dim=1).cpu().numpy()
        plt.plot(mean_error_per_time, 'r-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Mean error')
        plt.title('Error evolution over time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return true_traj, pred_traj


def display_evaluation_results_2d(
    evaluation_results,
    show_plots=True,
    channel: int = 0,
    interval_ms: int = 60,
    save_video_path: str | None = None,
    cmap: str = "seismic",
):
    """
    Display metrics and optional 2D animation for evaluation results.
    """
    true_traj = evaluation_results['true_traj']
    pred_traj = evaluation_results['pred_traj']

    print(f"Evaluation metrics (2D):")
    print(f"  - PSNR: {evaluation_results['psnr']:.3f} dB")
    print(f"  - SSIM: {evaluation_results['ssim']:.3f}")
    print(f"  - MSE: {evaluation_results['mse']:.6f}")
    print(f"  - R²: {evaluation_results['r2']:.4f}")

    if show_plots:
        true_path = None
        pred_path = None
        if save_video_path:
            root, ext = os.path.splitext(save_video_path)
            if not ext:
                ext = ".mp4"
            true_path = f"{root}_true{ext}"
            pred_path = f"{root}_pred{ext}"

        show_field(
            true_traj,
            title_prefix="True field",
            channel=channel,
            cmap=cmap,
            interval_ms=interval_ms,
            show_animation=True,
            save_video_path=true_path,
        )
        show_field(
            pred_traj,
            title_prefix="Predicted field",
            channel=channel,
            cmap=cmap,
            interval_ms=interval_ms,
            show_animation=True,
            save_video_path=pred_path,
        )

    return true_traj, pred_traj


def generate_model_predictions(model, train_loader, device, patch_radius,
                               verbose: bool = True, chunk_size: int = 3, nu_target=None, list_nus: bool = False, fallback_to_first: bool = True):
    """
    Version "confort" pour la visu : prend le premier sample du loader,
    appelle rollout_trajectory, et renvoie true_traj, pred_traj, nu.
    """
    # Small debugging helpers
    if list_nus:
        try:
            nus = get_loader_nu_values(train_loader)
            print(f"Loader contains {len(nus)} unique nu values: {nus}")
        except Exception as e:
            if verbose:
                print(f"Unable to enumerate loader nu values: {e}")

    # Pick a sample to visualize
    if nu_target is None:
        init_field, true_traj_batch, nu_batch = next(iter(train_loader))
    else:
        # try to find a batch where the first sample has the requested nu
        found = False
        for init_field, true_traj_batch, nu_batch in train_loader:
            try:
                this_nu = float(nu_batch[0].item())
            except Exception:
                this_nu = float(nu_batch)
            if abs(this_nu - float(nu_target)) < 1e-8:
                found = True
                break
        if not found:
            if fallback_to_first:
                if verbose:
                    print(f"No sample with nu={nu_target} found in the loader; falling back to the first sample.")
                init_field, true_traj_batch, nu_batch = next(iter(train_loader))
            else:
                raise ValueError(f"No sample with nu={nu_target} found in the loader")

    # Convert the picked batch into single trajectory and nu_value
    true_traj = true_traj_batch[0].to(device)   # (T, N)
    try:
        nu_value = float(nu_batch[0].item())
    except Exception:
        nu_value = float(nu_batch)

    if verbose:
        print(f"Generating predictions for nu = {nu_value:.4f}")
        print(f"True trajectory shape: {true_traj.shape}")
        print(f"Model type: {type(model).__name__}")

    pred_traj = rollout_trajectory(
        model=model,
        true_traj=true_traj,
        nu_value=nu_value,
        device=device,
        patch_radius=patch_radius,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    return true_traj, pred_traj, nu_value

def rollout_trajectory(
    model,
    true_traj: torch.Tensor,
    nu_value: float,
    device,
    patch_radius: int,
    chunk_size: int = 3,
    verbose: bool = False,
):
    """
    Reconstruit une trajectoire prédite à partir d'une trajectoire vraie.

    true_traj : (T, N) sur device CPU ou GPU
    nu_value  : scalaire (float)
    Retour:
      pred_traj : (T, N)
    """
    from training import build_patches_from_sequence, extract_spatial_patches

    true_traj = true_traj.to(device)
    T, N = true_traj.shape
    patch_size = 2 * patch_radius + 1

    model.eval()
    model_name = type(model).__name__

    # CNN 1-step : patch spatial (B, P) + nu
    is_improved_burgers = model_name == "ImprovedBurgersNet"
    is_cnn_patch = model_name in ["CNNControllerPatch", "CNNController"]
    # Tout le reste = modèles séquentiels (RNN, CNNHistory, SingleChannelSpaceTimeCNN, CNNSpaceTimeController, etc.)
    is_seq_model = not is_cnn_patch

    preds = []

    with torch.no_grad():
        if is_improved_burgers:
            # Autoregressive rollout: seed with the first window, then feed predictions.
            window_size = 4  # from __init__
            if T <= window_size:
                if verbose:
                    print(f"T={T} <= window_size={window_size}, on renvoie la vérité terrain.")
                pred_traj = true_traj.clone()
            else:
                pred_traj = [true_traj[:window_size, :]]
                for t in range(window_size, T):
                    if verbose and t % 10 == 0:
                        print(f"  ImprovedBurgers step {t}/{T}")

                    history = pred_traj[0][-window_size:, :].unsqueeze(0)  # (1, W, N)
                    pred_next = model(history).squeeze(1)  # (N,)
                    pred_traj.append(pred_next.unsqueeze(0))

                pred_traj = torch.cat(pred_traj, dim=0)[:T, :]
        elif is_seq_model:
            if T <= chunk_size:
                if verbose:
                    print(f"T={T} <= chunk_size={chunk_size}, on renvoie la vérité terrain.")
                pred_traj = true_traj.clone()
            else:
                # Autoregressive rollout: seed with the first chunk, then feed predictions.
                current_chunk = true_traj[:chunk_size, :].unsqueeze(0)  # (1, L, N)
                for t in range(chunk_size, T):
                    if verbose and t % 10 == 0:
                        print(f"  step {t}/{T}")

                    # (N, L, P)
                    patches = build_patches_from_sequence(
                        current_chunk, patch_radius, patch_size
                    )

                    nu_vals = torch.full(
                        (patches.size(0), 1),
                        nu_value,
                        device=device,
                    )

                    pred_next = model(patches, nu_vals)  # (N,)
                    preds.append(pred_next)

                    # Update chunk with predicted next step
                    next_step = pred_next.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
                    current_chunk = torch.cat([current_chunk[:, 1:, :], next_step], dim=1)

                pred_future = torch.stack(preds, dim=0)         # (T - chunk_size, N)
                pred_traj = torch.cat(
                    [true_traj[:chunk_size, :], pred_future],
                    dim=0
                )                                              # (T, N)

        else:
            # --- CNN 1-step (type CNNControllerPatch), autoregressive rollout
            pred_traj = [true_traj[0:1, :]]
            for t in range(1, T):
                if verbose and t % 10 == 0:
                    print(f"  step {t}/{T - 1}")

                field_t = pred_traj[-1]                        # (1, N)
                patches = extract_spatial_patches(field_t, patch_radius)  # (1, N, P)
                patches_flat = patches.reshape(N, -1)         # (N, P)

                nu_vals = torch.full((N, 1), nu_value, device=device)
                pred_next = model(patches_flat, nu_vals)      # (N,)
                pred_traj.append(pred_next.unsqueeze(0))

            pred_traj = torch.cat(pred_traj, dim=0)

    return pred_traj


def generate_model_predictions_2d(model, train_loader, device, patch_radius,
                                  verbose: bool = True, chunk_size: int = 3, nu_target=None, list_nus: bool = False, fallback_to_first: bool = True):
    """
    Version 2D pour la visu : prend un sample du loader,
    appelle rollout_trajectory_2d, et renvoie true_traj, pred_traj, nu.
    """
    # Small debugging helpers
    if list_nus:
        try:
            nus = get_loader_nu_values(train_loader)
            print(f"Loader contains {len(nus)} unique nu values: {nus}")
        except Exception as e:
            if verbose:
                print(f"Unable to enumerate loader nu values: {e}")

    # Pick a sample to visualize
    if nu_target is None:
        init_field, true_traj_batch, nu_batch = next(iter(train_loader))
    else:
        # try to find a batch where the first sample has the requested nu
        found = False
        for init_field, true_traj_batch, nu_batch in train_loader:
            try:
                this_nu = float(nu_batch[0].item())
            except Exception:
                this_nu = float(nu_batch)
            if abs(this_nu - float(nu_target)) < 1e-8:
                found = True
                break
        if not found:
            if fallback_to_first:
                if verbose:
                    print(f"No sample with nu={nu_target} found in the loader; falling back to the first sample.")
                init_field, true_traj_batch, nu_batch = next(iter(train_loader))
            else:
                raise ValueError(f"No sample with nu={nu_target} found in the loader")

    # Convert the picked batch into single trajectory and nu_value
    true_traj = true_traj_batch[0].to(device)   # (T, C, H, W)
    try:
        nu_value = float(nu_batch[0].item())
    except Exception:
        nu_value = float(nu_batch)

    if verbose:
        print(f"Generating 2D predictions for nu = {nu_value:.4f}")
        print(f"True trajectory shape: {true_traj.shape}")
        print(f"Model type: {type(model).__name__}")

    pred_traj = rollout_trajectory_2d(
        model=model,
        true_traj=true_traj,
        nu_value=nu_value,
        device=device,
        patch_radius=patch_radius,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    return true_traj, pred_traj, nu_value


def rollout_trajectory_2d(
    model,
    true_traj: torch.Tensor,
    nu_value: float,
    device,
    patch_radius: int,
    chunk_size: int = 3,
    verbose: bool = False,
):
    """
    Reconstruit une trajectoire prédite à partir d'une trajectoire vraie (2D).

    true_traj : (T, C, H, W)
    nu_value  : scalaire (float)
    Retour:
      pred_traj : (T, C, H, W)
    """
    from training import build_patches_from_sequence_2d

    true_traj = true_traj.to(device)
    T, C, H, W = true_traj.shape

    model.eval()
    preds = []

    with torch.no_grad():
        if T <= chunk_size:
            if verbose:
                print(f"T={T} <= chunk_size={chunk_size}, on renvoie la vérité terrain.")
            pred_traj = true_traj.clone()
        else:
            for t in range(T - chunk_size):
                if verbose and t % 10 == 0:
                    print(f"  step {t}/{T - chunk_size}")

                # (1, L, C, H, W)
                current_chunk = true_traj[t:t+chunk_size].unsqueeze(0)

                # patches: (1, L, H, W, P)
                patches = build_patches_from_sequence_2d(current_chunk, patch_radius)
                # global pooling -> (1, L, P)
                patch_seq = patches.mean(dim=(2, 3))

                nu_vals = torch.full((1, 1), nu_value, device=device)
                pred_next = model(patch_seq, nu_vals)  # (1, C, H, W)
                preds.append(pred_next.squeeze(0))

            pred_future = torch.stack(preds, dim=0)           # (T - chunk_size, C, H, W)
            pred_traj = torch.cat([true_traj[:chunk_size], pred_future], dim=0)

    return pred_traj

# ==========================================
# 4. Evaluation Metrics
# ==========================================

class Metrics:
    def __init__(self, dx=DX, dt=DT, nu=0.0):
        self.dx = dx
        self.dt = dt
        self.nu = nu
        self.zero_tolerance = 0.1 
    
    def mse(self, pred, gt):
        return np.mean((pred - gt)**2)
    
    def relative_l2(self, pred, gt):
        return np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-10)
    
    def correlation_per_timestep(self, pred, gt):
        correlations = []
        for t in range(len(pred)):
            if np.std(pred[t]) < 1e-10 or np.std(gt[t]) < 1e-10:
                correlations.append(0.0)
            else:
                corr, _ = stats.pearsonr(pred[t].flatten(), gt[t].flatten())
                correlations.append(corr)
        return np.array(correlations)
    
    def mass(self, u):
        return np.sum(u, axis=-1) * self.dx
    
    def mass_conservation_error(self, trajectory):
        mass_t = self.mass(trajectory)
        initial_mass = mass_t[0]
        
        if np.abs(initial_mass) < self.zero_tolerance:
            return np.max(np.abs(mass_t - initial_mass))
        else:
            return np.max(np.abs(mass_t - initial_mass) / np.abs(initial_mass))
    
    def energy(self, u):
        return 0.5 * np.sum(u**2, axis=-1) * self.dx
    
    def check_energy_monotonicity(self, trajectory):
        energy_t = self.energy(trajectory)
        energy_diff = np.diff(energy_t)
        # Check percentage of steps where energy does not increase
        return np.mean(energy_diff <= 1e-6)
    


# ==========================================
# 5. Evaluation & Main Execution
# ==========================================

def evaluate_model(model, test_loader, window_size, T_test):
    model.eval()
    device = next(model.parameters()).device
    all_metrics = {}

    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    with torch.no_grad():
        for sample_idx, (initial_field, true_trajectory, viscosity_val) in enumerate(test_loader):
            true_trajectory = true_trajectory.to(device).squeeze(0) # (T, N)
            
            nu = viscosity_val.item()
            metrics = Metrics(nu=nu)
            
            init_window_gt = true_trajectory[:window_size, :].unsqueeze(0) # (1, W, N)
            
            pred_trajectory = []
            current_window = init_window_gt
            
            # --- Rollout Prediction ---
            for t in range(T_test - window_size):
                pred = model(current_window) # (1, N)
                pred_trajectory.append(pred)
                
                # Recurrent update: shift and append
                current_window = torch.cat([current_window[:, 1:, :], pred.unsqueeze(1)], dim=1)
            
            pred_trajectory_2d = torch.stack(pred_trajectory, dim=0).squeeze(1) # (T-W, N)
            
            gt_aligned = true_trajectory[window_size:].cpu().numpy()
            pred_aligned = pred_trajectory_2d.cpu().numpy()
            
            # --- Compute Metrics ---
            mse_val = metrics.mse(pred_aligned, gt_aligned)
            rel_l2 = metrics.relative_l2(pred_aligned, gt_aligned)
            mass_err = metrics.mass_conservation_error(pred_aligned)
            e_mono_pct = metrics.check_energy_monotonicity(pred_aligned) * 100
            
            correlations = metrics.correlation_per_timestep(pred_aligned, gt_aligned)
            t_steps = len(correlations)
            t_corr_9 = np.where(correlations < 0.9)[0]
            t_corr_09 = t_corr_9[0] if len(t_corr_9) > 0 else t_steps

            all_metrics[f'Sample_{sample_idx}_nu{nu:.3f}'] = {
                'MSE': mse_val,
                'Rel L2': rel_l2,
                'Mass Abs Error': mass_err,
                'E Mono %': e_mono_pct,
                'T_corr<0.9': t_corr_09,
            }
            
            if sample_idx == 0:
                print(f"Sample {sample_idx} (nu={nu:.3f}) Results:")
                print(f"  MSE: {mse_val:.4e} | Rel L2: {rel_l2:.4f}")
                print(f"  Max Mass Abs Error (Corrected): {mass_err:.4e}")
                print(f"  Energy Monotonicity: {e_mono_pct:.1f}%")
                print(f"  Time to Corr < 0.9: {t_corr_09} steps")
                
                full_gt = true_trajectory.cpu().numpy()
                full_pred = np.concatenate((full_gt[:window_size], pred_aligned), axis=0)

                plot_learning_progress(
                    torch.from_numpy(full_gt), 
                    torch.from_numpy(full_pred), 
                    'Final', sample_idx
                )
    
    print("\nEvaluation Complete.")
    return all_metrics

# ----------------------- Akash model's evaluation ---------------------------
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

# # ==========================================
# # 4. Corrected Evaluation Metrics
# # ==========================================
# class BurgersMetrics:
#     """Evaluation metrics with corrected mass handling"""
    
#     def __init__(self, dx, dt, nu):
#         self.dx = dx
#         self.dt = dt
#         self.nu = nu
    
#     def mse(self, pred, gt):
#         return np.mean((pred - gt)**2)
    
#     def relative_l2(self, pred, gt):
#         return np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-10)
    
#     def relative_l2_per_timestep(self, pred, gt):
#         num = np.sqrt(np.sum((pred - gt)**2, axis=-1))
#         den = np.sqrt(np.sum(gt**2, axis=-1))
#         return num / (den + 1e-10)
    
#     def psnr(self, pred, gt):
#         mse_val = self.mse(pred, gt)
#         data_range = gt.max() - gt.min()
#         if mse_val < 1e-10:
#             return float('inf')
#         return 20 * np.log10(data_range / np.sqrt(mse_val))
    
#     def ssim_score(self, pred, gt):
#         if not HAS_SKIMAGE:
#             return 0.0
#         data_range = gt.max() - gt.min()
#         return ssim(gt, pred, data_range=data_range)
    
#     def correlation_per_timestep(self, pred, gt):
#         correlations = []
#         for t in range(len(pred)):
#             if np.std(pred[t]) < 1e-10 or np.std(gt[t]) < 1e-10:
#                 correlations.append(0.0)
#             else:
#                 corr, _ = stats.pearsonr(pred[t].flatten(), gt[t].flatten())
#                 correlations.append(corr)
#         return np.array(correlations)
    
#     def time_to_threshold(self, values, threshold):
#         crossed = np.where(values < threshold)[0]
#         return crossed[0] if len(crossed) > 0 else len(values)
    
#     def is_stable(self, trajectory, bounds=(-5, 5)):
#         return np.all(np.abs(trajectory) < bounds[1])
    
#     def has_nan_or_inf(self, trajectory):
#         return np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory))
    
#     # CORRECTED: Mass metric
#     def mass(self, u):
#         """Mass: M = ∫u dx"""
#         return np.sum(u, axis=-1) * self.dx
    
#     def mass_conservation_error(self, trajectory):
#         """
#         CORRECTED: Use absolute mass change when initial mass is near zero.
#         For sin(x) initial condition, ∫sin(x)dx = 0 over [0, 2π].
#         """
#         mass_t = self.mass(trajectory)
#         initial_mass = mass_t[0]
        
#         # If initial mass is near zero, use absolute change
#         if np.abs(initial_mass) < 0.1:
#             return np.abs(mass_t - initial_mass)  # Absolute, not relative
#         else:
#             return np.abs(mass_t - initial_mass) / np.abs(initial_mass)
    
#     def energy(self, u):
#         """Energy: E = ½∫u² dx"""
#         return 0.5 * np.sum(u**2, axis=-1) * self.dx
    
#     def energy_dissipation_error(self, pred, gt):
#         pred_energy = self.energy(pred)
#         gt_energy = self.energy(gt)
#         return np.abs(pred_energy - gt_energy) / (gt_energy + 1e-10)
    
#     def check_energy_monotonicity(self, trajectory):
#         energy_t = self.energy(trajectory)
#         energy_diff = np.diff(energy_t)
#         return np.mean(energy_diff <= 1e-6)  # Allow small numerical tolerance
    
#     def pde_residual(self, u_curr, u_next):
#         u_t = (u_next - u_curr) / self.dt
#         u_left = np.roll(u_curr, 1, axis=-1)
#         u_right = np.roll(u_curr, -1, axis=-1)
#         u_x = (u_right - u_left) / (2 * self.dx)
#         u_xx = (u_right - 2*u_curr + u_left) / self.dx**2
#         residual = u_t + u_curr * u_x - self.nu * u_xx
#         return residual
    
#     def mean_pde_residual(self, trajectory):
#         residuals = []
#         for t in range(len(trajectory) - 1):
#             res = self.pde_residual(trajectory[t], trajectory[t+1])
#             residuals.append(np.mean(np.abs(res)))
#         return np.mean(residuals)
    
#     def energy_spectrum(self, u):
#         fft_u = np.fft.fft(u, axis=-1)
#         spectrum = np.abs(fft_u)**2
#         n = spectrum.shape[-1]
#         return spectrum[..., :n//2]
    
#     def spectrum_error(self, pred, gt):
#         pred_spec = self.energy_spectrum(pred)
#         gt_spec = self.energy_spectrum(gt)
#         # Use log ratio for better scaling
#         log_pred = np.log10(pred_spec + 1e-10)
#         log_gt = np.log10(gt_spec + 1e-10)
#         return np.mean(np.abs(log_pred - log_gt))
    
#     def gradient_error(self, pred, gt):
#         pred_grad = np.gradient(pred, self.dx, axis=-1)
#         gt_grad = np.gradient(gt, self.dx, axis=-1)
#         return np.mean(np.abs(pred_grad - gt_grad), axis=-1)
    
#     def max_gradient(self, u):
#         u_grad = np.gradient(u, self.dx, axis=-1)
#         return np.max(np.abs(u_grad), axis=-1)



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