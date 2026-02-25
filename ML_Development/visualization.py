# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch


def show_field(
    field,
    title_prefix: str = "Field",
    channel: int = 0,
    cmap: str = "seismic",
    interval_ms: int = 60,
    show_animation: bool = True,
    save_video_path: str | None = None,
    x=None,
    y=None,
    t=None,
    label: str = "u(x,y)",
    vmin=None,
    vmax=None,
    origin: str = "lower",
):
    """
    Display an animation of a 2D field over time.

    field: (T, H, W) or (T, C, H, W) numpy array or torch tensor.
    """
    if isinstance(field, torch.Tensor):
        field_np = field.detach().cpu().numpy()
    else:
        field_np = np.asarray(field)

    if field_np.ndim == 4:
        T, C, H, W = field_np.shape
        if channel < 0 or channel >= C:
            raise ValueError(f"Invalid channel index {channel}. Available: 0..{C-1}.")
        data = field_np[:, channel, :, :]
    elif field_np.ndim == 3:
        T, H, W = field_np.shape
        data = field_np
    elif field_np.ndim == 2:
        H, W = field_np.shape
        data = field_np[None, :, :]
        T = 1
    else:
        raise ValueError("show_field expects (T, H, W) or (T, C, H, W) data.")

    if vmin is None:
        vmin = float(np.min(data))
    if vmax is None:
        vmax = float(np.max(data))

    extent = None
    if x is not None and y is not None:
        extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        data[0].T,
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
        origin=origin,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    time_label = "0" if t is None else f"{t[0]:.3f}"
    title = ax.set_title(f"{title_prefix} — t={time_label}")
    plt.colorbar(im, ax=ax, label=label)

    def update(frame):
        im.set_data(data[frame].T)
        if t is None:
            title.set_text(f"{title_prefix} — t={frame}")
        else:
            title.set_text(f"{title_prefix} — t={t[frame]:.3f}")
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=interval_ms, blit=False
    )

    if save_video_path:
        ext = os.path.splitext(save_video_path)[1].lower()
        fps = max(1, int(1000 / interval_ms))
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps)
        anim.save(save_video_path, writer=writer, dpi=150)
        print(f"✅ Vidéo sauvegardée: {save_video_path}")

    if show_animation:
        plt.show()
    else:
        plt.close(fig)

    return anim

def plot_trajectories(true_traj: torch.Tensor, pred_traj: torch.Tensor, title_suffix: str = ""):
    """
    true_traj, pred_traj : (T, N) (single trajectory)
    """
    true_np = true_traj.cpu().numpy().T   # (N, T)
    pred_np = pred_traj.cpu().numpy().T
    err_np  = (true_np - pred_np) ** 2

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(true_np, aspect="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("time")
    plt.ylabel("space")
    plt.title("True" + title_suffix)

    plt.subplot(1, 3, 2)
    plt.imshow(pred_np, aspect="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("time")
    plt.ylabel("space")
    plt.title("Predicted" + title_suffix)

    plt.subplot(1, 3, 3)
    plt.imshow(err_np, aspect="auto", cmap="inferno")
    plt.colorbar(label="squared error")
    plt.xlabel("time")
    plt.ylabel("space")
    plt.title("Error")

    plt.tight_layout()
    plt.show()


def plot_trajectories_2d(
    true_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    title_suffix: str = "",
    channel: int = 0,
    cmap: str = "seismic",
    interval_ms: int = 60,
    show_animation: bool = True,
    save_video_path: str | None = None,
    error_cmap: str = "inferno",
):
    """
    true_traj, pred_traj : (T, C, H, W)
    Creates a 2D animation for a given channel.
    """
    true_np = true_traj.detach().cpu().numpy()
    pred_np = pred_traj.detach().cpu().numpy()

    if true_np.ndim != 4 or pred_np.ndim != 4:
        raise ValueError("plot_trajectories_2d expects (T, C, H, W) tensors.")

    T, C, H, W = true_np.shape
    if channel < 0 or channel >= C:
        raise ValueError(f"Invalid channel index {channel}. Available: 0..{C-1}.")

    true_c = true_np[:, channel, :, :]
    pred_c = pred_np[:, channel, :, :]
    err_c = (true_c - pred_c) ** 2

    vmin = float(min(true_c.min(), pred_c.min()))
    vmax = float(max(true_c.max(), pred_c.max()))

    base_path = save_video_path
    true_path = None
    pred_path = None
    err_path = None
    if base_path:
        root, ext = os.path.splitext(base_path)
        if not ext:
            ext = ".mp4"
        true_path = f"{root}_true{ext}"
        pred_path = f"{root}_pred{ext}"
        err_path = f"{root}_err{ext}"

    show_field(
        true_c,
        title_prefix="True" + title_suffix,
        channel=0,
        cmap=cmap,
        interval_ms=interval_ms,
        show_animation=show_animation,
        save_video_path=true_path,
        vmin=vmin,
        vmax=vmax,
    )
    show_field(
        pred_c,
        title_prefix="Predicted" + title_suffix,
        channel=0,
        cmap=cmap,
        interval_ms=interval_ms,
        show_animation=show_animation,
        save_video_path=pred_path,
        vmin=vmin,
        vmax=vmax,
    )
    show_field(
        err_c,
        title_prefix="Squared error" + title_suffix,
        channel=0,
        cmap=error_cmap,
        interval_ms=interval_ms,
        show_animation=show_animation,
        save_video_path=err_path,
    )


def combine_gifs_horizontal(
    gif_paths,
    output_path,
    *,
    background_color=(0, 0, 0, 0),
):
    """
    Combine multiple GIFs horizontally into a single animated GIF.
    gif_paths: list of input GIF paths in left-to-right order.
    output_path: output GIF path.
    background_color: RGBA tuple for padding background.
    """
    try:
        from PIL import Image, ImageSequence
    except Exception as exc:
        raise ImportError("Pillow is required. Install with `pip install pillow`.") from exc

    if not gif_paths:
        raise ValueError("combine_gifs_horizontal expects at least one input GIF path.")

    images = [Image.open(p) for p in gif_paths]

    # --- Extract frames as *copies* + per-frame durations (critical for animation correctness)
    frames_lists = []
    durations_lists = []
    for img in images:
        frames = []
        durs = []
        for fr in ImageSequence.Iterator(img):
            fr_copy = fr.copy()  # <-- critical: detach from underlying img buffer
            frames.append(fr_copy)
            durs.append(int(fr.info.get("duration", img.info.get("duration", 100))))
        frames_lists.append(frames)
        durations_lists.append(durs)

    frame_count = min(len(frames) for frames in frames_lists)
    if frame_count <= 1:
        # If min is 1, output will look static. This is often expected if one input has 1 frame.
        # Still, we save as animated only if there are >=2 frames.
        raise ValueError(
            f"Not enough frames to animate (min frame count across inputs = {frame_count}). "
            "At least one input GIF likely has only 1 frame."
        )

    # Choose a sane default duration (median of first-frame durations)
    first_durs = [dl[0] if dl else 100 for dl in durations_lists]
    first_durs_sorted = sorted(first_durs)
    mid = len(first_durs_sorted) // 2
    default_duration = first_durs_sorted[mid] if first_durs_sorted else 100

    combined_frames = []
    combined_durations = []

    for i in range(frame_count):
        # Convert each source frame to RGBA (safe compositing)
        src_frames = []
        for idx in range(len(frames_lists)):
            src_frames.append(frames_lists[idx][i].convert("RGBA"))

        max_h = max(f.height for f in src_frames)
        total_w = sum(f.width for f in src_frames)

        canvas = Image.new("RGBA", (total_w, max_h), background_color)

        x_offset = 0
        for f in src_frames:
            canvas.paste(f, (x_offset, 0), f)  # use alpha mask
            x_offset += f.width

        combined_frames.append(canvas)

        # duration policy: take duration from the first GIF's frame i (fallback to default)
        dur0 = durations_lists[0][i] if i < len(durations_lists[0]) else default_duration
        combined_durations.append(int(dur0))

    # --- Palette handling: use a single global palette to keep Pillow happy & consistent
    # Quantize first frame, then quantize others using the first frame palette.
    pal0 = combined_frames[0].convert("P", palette=Image.ADAPTIVE, colors=256)
    paletted = [pal0]
    for fr in combined_frames[1:]:
        paletted.append(fr.quantize(palette=pal0))

    paletted[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=paletted[1:],
        duration=combined_durations,
        loop=0,
        disposal=2,
        optimize=False,  # avoid collapsing frames unexpectedly
    )



def plot_losses(train_losses, test_losses=None, title: str = "Loss per epoch"):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, marker="o", label="train")
    if test_losses is not None:
        plt.plot(test_losses, marker="s", label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_space_time_kernel(
    model,
    layer_name: str = "conv1",
    channel_type: str = "field",
    out_channel: int | None = None,
    cmap: str = "bwr",
):
    """
    Visualise le noyau spatio-temporel d'une couche conv2d (par défaut `conv1`)
    pour des modèles comme:
      - SingleChannelSpaceTimeCNN (conv1 : 2 -> 1, kernel 3x3)
      - CNNSpaceTimeController    (conv1 : 2 -> C, kernel k_t x k_x)

    - model        : modèle PyTorch (doit avoir un attribut layer_name)
    - layer_name   : nom de la couche, ex. "conv1" ou "conv2"
    - channel_type : "field" (u) ou "nu" (viscosité)
    - out_channel  :
        * None -> moyenne sur tous les filtres de sortie
        * int  -> un filtre de sortie spécifique
    """

    if not hasattr(model, layer_name):
        print(
            f"plot_space_time_kernel: le modèle n'a pas de couche '{layer_name}'. "
            "Rien à afficher."
        )
        return

    layer = getattr(model, layer_name)
    if not hasattr(layer, "weight"):
        print(
            f"plot_space_time_kernel: la couche '{layer_name}' n'a pas d'attribut 'weight'. "
            "Rien à afficher."
        )
        return

    weight = layer.weight.detach().cpu().numpy()
    # On ne traite que les conv 2D: (C_out, C_in, k_t, k_x)
    if weight.ndim != 4:
        print(
            f"plot_space_time_kernel: '{layer_name}.weight' a une shape {weight.shape}, "
            "attendu: 4D (C_out, C_in, k_t, k_x). Modèle ignoré."
        )
        return

    C_out, C_in, k_t, k_x = weight.shape

    # Sélection du canal d'entrée: champ ou viscosité
    if channel_type == "field":
        in_idx = 0
    elif channel_type == "nu":
        if C_in < 2:
            print(
                f"plot_space_time_kernel: channel_type='nu' demandé mais la couche "
                f"'{layer_name}' n'a que {C_in} canal(x) d'entrée. "
                "On affiche le canal 0 à la place."
            )
            in_idx = 0
        else:
            in_idx = 1
    else:
        raise ValueError("channel_type doit être 'field' ou 'nu'.")

    if in_idx >= C_in:
        print(
            f"plot_space_time_kernel: index de canal d'entrée {in_idx} "
            f">= C_in={C_in}. Rien à afficher."
        )
        return

    # Sélection du filtre de sortie
    if out_channel is None:
        # moyenne des noyaux sur tous les filtres de sortie
        kernel = weight[:, in_idx, :, :].mean(axis=0)  # (k_t, k_x)
        title = (
            f"{layer_name} – {channel_type} kernel "
            f"(mean over {C_out} filters)"
        )
    else:
        if not (0 <= out_channel < C_out):
            print(
                f"plot_space_time_kernel: out_channel={out_channel} hors bornes "
                f"[0, {C_out-1}]. Rien à afficher."
            )
            return
        kernel = weight[out_channel, in_idx, :, :]      # (k_t, k_x)
        title = (
            f"{layer_name} – {channel_type} kernel – filter {out_channel}"
        )

    # Plot
    t_offsets = np.arange(k_t) - k_t // 2
    x_offsets = np.arange(k_x) - k_x // 2

    plt.figure(figsize=(5, 4))
    plt.imshow(kernel, cmap=cmap, aspect="auto", origin="lower")
    plt.colorbar(label="weight")
    plt.xticks(np.arange(k_x), x_offsets)
    plt.yticks(np.arange(k_t), t_offsets)
    plt.xlabel("space offset (Δx)")
    plt.ylabel("time offset (Δt)")
    plt.title(title)
    plt.tight_layout()
    plt.show()




def plot_learning_progress(true_traj, pred_traj, epoch, sample_idx=0):
    true_np = true_traj.cpu().numpy().T  # (space, time)
    pred_np = pred_traj.cpu().numpy().T
    # error_np = (true_np - pred_np) ** 2
    error_np = abs(true_np - pred_np)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(true_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('Time t')
    plt.ylabel('Position x')
    plt.title(f'True Trajectory (Sample {sample_idx})')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('Time t')
    plt.ylabel('Position x')
    plt.title(f'Predicted Trajectory (Epoch {epoch})')

    plt.subplot(1, 3, 3)
    plt.imshow(error_np, aspect='auto', cmap='inferno')
    plt.colorbar(label='error')
    plt.xlabel('Time t')
    plt.ylabel('Position x')
    plt.title('Absolute Error')

    plt.tight_layout()
    plt.show()
