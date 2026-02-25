import os, random, collections.abc, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


class Grid2D:
    def __init__(self, nb_points_x, x_min, x_max, nb_points_y, y_min, y_max, t_max, dt):
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        self.nb_points_x = int(nb_points_x)
        self.nb_points_y = int(nb_points_y)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.dt = float(dt)

        # temps
        self.nb_points_t = int(np.floor(t_max / dt)) + 1
        self.t_max = self.dt * (self.nb_points_t - 1)

        # espace
        self.dx = (self.x_max - self.x_min) / (self.nb_points_x - 1)
        self.dy = (self.y_max - self.y_min) / (self.nb_points_y - 1)
        self.x = np.linspace(self.x_min, self.x_max, self.nb_points_x)
        self.y = np.linspace(self.y_min, self.y_max, self.nb_points_y)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.t = np.linspace(0, self.t_max, self.nb_points_t)

    def save_npz(self, path, U, *, nu=None, speed=None, tag=""):
        meta = dict(
            U=U.astype(np.float32),
            x=self.x.astype(np.float32),
            y=self.y.astype(np.float32),
            t=self.t.astype(np.float32),
            dx=np.float32(self.dx),
            dy=np.float32(self.dy),
            dt=np.float32(self.dt),
        )
        if nu is not None:
            meta["nu"] = np.float32(nu)
        if speed is not None:
            meta["speed"] = np.float32(speed)
        if tag:
            meta["tag"] = np.str_(tag)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        np.savez_compressed(path, **meta)

    # (Optionnel) petites aides pour viz rapide
    def plot(self, U, title="Burgers 2D", time_step=-1, label=None, cmap='seismic', save_path=None):
        """Affiche (ou sauvegarde) U à un pas de temps donné (par défaut le dernier)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Condition initiale
        im1 = ax1.imshow(U[0, :, :].T, extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
                        cmap=cmap, interpolation='nearest', aspect='auto', origin='lower')
        ax1.set_title(f'{title} - t=0')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, label=label)
        
        # État final ou au temps demandé
        im2 = ax2.imshow(U[time_step, :, :].T, extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
                        cmap=cmap, interpolation='nearest', aspect='auto', origin='lower')
        ax2.set_title(f'{title} - t={self.t[time_step]:.3f}')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2, label=label)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot sauvegardé: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_initial_condition(self, U, title="U(x, y, t=0)", label=None, cmap='seismic', save_path=None):
        """Affiche (ou sauvegarde) la condition initiale 2D"""
        # accepte U 3D (t,x,y) ou 2D (x,y)
        U0 = U[0, :, :] if getattr(U, "ndim", 2) == 3 else U
        plt.figure(figsize=(10, 8))
        im = plt.imshow(U0.T, extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
                       cmap=cmap, interpolation='nearest', aspect='auto', origin='lower')
        plt.colorbar(label=label)
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot sauvegardé: {save_path}")
            plt.close()
        else:
            plt.show()


def sample_initial_condition_burgers_2d(X, Y, speed, kind="shock"):
    """Génère des conditions initiales 2D pour l'équation de Burger"""
    if kind == "shock":
        u0 = np.where(X < 0, speed, -speed)
    elif kind == "shock_with_gap":
        u0 = np.where((X >= -1.0) & (X <= 1.0), 0.0, np.where(X < 0, 1.0, -1.0)) * speed
    elif kind == "rarefaction":
        u0 = np.where(X < 0, -speed, speed)
    elif kind == "hyperbolic_tangent":
        u0 = np.tanh(X) * speed
    elif kind == "sine":
        kx = np.random.randint(1, 5)
        ky = np.random.randint(1, 5)
        phase_x = np.random.uniform(0, 2*np.pi)
        phase_y = np.random.uniform(0, 2*np.pi)
        xmax = max(1e-12, X.max())
        ymax = max(1e-12, Y.max())
        u0 = (np.sin(2*np.pi*kx*X/xmax + phase_x) * np.sin(2*np.pi*ky*Y/ymax + phase_y)) * speed
    elif kind == "radial":
        # Nouveau type: onde radiale
        r = np.sqrt((X - X.mean())**2 + (Y - Y.mean())**2)
        u0 = np.exp(-r**2) * speed
    else:  # "smooth" aléatoire
        u0 = (np.random.rand(*X.shape) * 2 - 1) * speed
        # Lissage 2D simple
        u0_smooth = np.zeros_like(u0)
        u0_smooth[1:-1, 1:-1] = (u0[1:-1, 1:-1] + u0[:-2, 1:-1] + u0[2:, 1:-1] + 
                                u0[1:-1, :-2] + u0[1:-1, 2:]) / 5.0
        u0_smooth[0, :] = u0[0, :]
        u0_smooth[-1, :] = u0[-1, :]
        u0_smooth[:, 0] = u0[:, 0]
        u0_smooth[:, -1] = u0[:, -1]
        u0 = u0_smooth
    return u0

def make_initial_condition_burgers_fn_2d(X, Y, speed, kind=None):
    if kind is None:
        kind = np.random.choice(["shock", "rarefaction", "sine", "smooth", "radial"])
    u0 = sample_initial_condition_burgers_2d(X, Y, speed, kind)
    return (lambda _X, _Y: u0), kind


def bc_periodic_2d(u):
    """Conditions aux limites périodiques pour 2D"""
    nx, ny = u.shape
    up = np.zeros((nx + 2, ny + 2), dtype=u.dtype)
    up[1:-1, 1:-1] = u
    # Périodique en x
    up[0, 1:-1] = u[-1, :]
    up[-1, 1:-1] = u[0, :]
    # Périodique en y
    up[1:-1, 0] = u[:, -1]
    up[1:-1, -1] = u[:, 0]
    # Coins
    up[0, 0] = u[-1, -1]; up[0, -1] = u[-1, 0]
    up[-1, 0] = u[0, -1]; up[-1, -1] = u[0, 0]
    return up


def bc_neumann_zero_2d(u):
    """Conditions aux limites de Neumann nulles pour 2D"""
    nx, ny = u.shape
    up = np.zeros((nx + 2, ny + 2), dtype=u.dtype)
    up[1:-1, 1:-1] = u
    # Neumann en x
    up[0, 1:-1] = u[0, :]
    up[-1, 1:-1] = u[-1, :]
    # Neumann en y
    up[1:-1, 0] = u[:, 0]
    up[1:-1, -1] = u[:, -1]
    # Coins
    up[0, 0] = u[0, 0]; up[0, -1] = u[0, -1]
    up[-1, 0] = u[-1, 0]; up[-1, -1] = u[-1, -1]
    return up


class Burgers2D:
    def __init__(self, grid, nu, initial_condition, boundary_condition=bc_periodic_2d, cfl_safety=0.5):
        self.grid = grid
        self.nu = float(nu)
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        self.cfl_safety = float(cfl_safety)

    @staticmethod
    def rusanov_flux(uL, uR):
        """Flux de Rusanov pour l'équation de Burger"""
        lam = np.maximum(np.abs(uL), np.abs(uR))
        return 0.5 * (0.5*uL*uL + 0.5*uR*uR) - 0.5 * lam * (uR - uL)

    def laplacian_2d(self, U):
        """Calcul du Laplacien 2D avec différences finies"""
        nx, ny = U.shape
        lapl = np.zeros_like(U)
        
        # Dérivées secondes en x
        if nx >= 3:
            lapl[1:-1, :] += (U[2:, :] - 2*U[1:-1, :] + U[:-2, :]) / (self.grid.dx ** 2)
        if nx >= 2:
            lapl[0, :] += (U[1, :] - 2*U[0, :] + U[0, :]) / (self.grid.dx ** 2)
            lapl[-1, :] += (U[-1, :] - 2*U[-1, :] + U[-2, :]) / (self.grid.dx ** 2)
            
        # Dérivées secondes en y
        if ny >= 3:
            lapl[:, 1:-1] += (U[:, 2:] - 2*U[:, 1:-1] + U[:, :-2]) / (self.grid.dy ** 2)
        if ny >= 2:
            lapl[:, 0] += (U[:, 1] - 2*U[:, 0] + U[:, 0]) / (self.grid.dy ** 2)
            lapl[:, -1] += (U[:, -1] - 2*U[:, -1] + U[:, -2]) / (self.grid.dy ** 2)
            
        return lapl

    def _apply_boundary(self, u):
        """Applique les conditions aux limites 2D"""
        if self.boundary_condition is None:
            # Conditions de Neumann par défaut
            return bc_neumann_zero_2d(u)
        padded = self.boundary_condition(u)
        return np.asarray(padded)

    def check_cfl_burgers_2d(self, u_now):
        """Vérification CFL pour 2D"""
        dx = abs(float(self.grid.dx))
        dy = abs(float(self.grid.dy))
        dt = float(self.grid.dt)
        nu = float(self.nu)
        umax = float(np.max(np.abs(u_now))) + 1e-14
        
        # Limite convective (plus restrictive entre x et y)
        conv_limit_x = dx / umax
        conv_limit_y = dy / umax
        conv_limit = min(conv_limit_x, conv_limit_y)
        
        # Limite diffusive (plus restrictive entre x et y)
        diff_limit_x = dx * dx / (2.0 * nu + 1e-14)
        diff_limit_y = dy * dy / (2.0 * nu + 1e-14)
        diff_limit = min(diff_limit_x, diff_limit_y)
        
        limit = self.cfl_safety * min(conv_limit, diff_limit)
        if dt > limit:
            raise ValueError(
                "Unstable explicit step:\n"
                f"  dt={dt:.3e} > safety*min(dx/|u|, dy/|u|, dx²/(2ν), dy²/(2ν))={limit:.3e}\n"
                f"  (dx={dx:.3e}, dy={dy:.3e}, ν={nu:.3e}, max|u|={umax:.3e}, safety={self.cfl_safety:.2f})"
            )

    def simulate(self):
        """Simulation de l'équation de Burger 2D + temps"""
        nt = len(self.grid.t)
        nx = len(self.grid.x)
        ny = len(self.grid.y)
        dx = float(self.grid.dx)
        dy = float(self.grid.dy)
        dt = float(self.grid.dt)
        nu = float(self.nu)

        U = np.zeros((nt, nx, ny))
        U[0, :, :] = self.initial_condition(self.grid.X, self.grid.Y)

        self.check_cfl_burgers_2d(U[0, :, :])

        for n in range(nt - 1):
            u = U[n, :, :]

            # Application des conditions aux limites
            up = self._apply_boundary(u)  # taille (nx+2, ny+2)
            
            # Calcul des flux en x
            uL_x = up[:-1, 1:-1]  # (nx+1, ny)
            uR_x = up[1:, 1:-1]   # (nx+1, ny)
            Fh_x = self.rusanov_flux(uL_x, uR_x)
            conv_x = -(Fh_x[1:, :] - Fh_x[:-1, :]) / dx
            
            # Calcul des flux en y
            uL_y = up[1:-1, :-1]  # (nx, ny+1)
            uR_y = up[1:-1, 1:]   # (nx, ny+1)
            Fh_y = self.rusanov_flux(uL_y, uR_y)
            conv_y = -(Fh_y[:, 1:] - Fh_y[:, :-1]) / dy
            
            # Terme de convection total
            conv = conv_x + conv_y

            # Terme de diffusion (Laplacien)
            diff = nu * self.laplacian_2d(u)

            # Mise à jour d'Euler explicite
            U[n + 1, :, :] = u + dt * (conv + diff)

            self.check_cfl_burgers_2d(U[n + 1, :, :])

        return U


def _tolist(x):
    return list(x) if (isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))) else [x]

def make_grid_2d(nbx, x_min, x_max, nby, y_min, y_max, dt, *, n_steps=None, t_final=None):
    """Crée une grille 2D spatiale + temps"""
    if t_final is None:
        if n_steps is None:
            raise ValueError("Spécifie n_steps ou t_final.")
        t_final = (int(n_steps) - 1) * dt
    return Grid2D(int(nbx), float(x_min), float(x_max), int(nby), float(y_min), float(y_max), float(t_final), float(dt))

def run_one_sim_burgers_2d(grid, nu, speed, ic_kind=None, cfl_safety=0.5, max_retries=4, boundary_condition=bc_periodic_2d):
    """Lance une simulation Burger 2D avec gestion des erreurs"""
    dt = grid.dt
    nbx = grid.nb_points_x
    nby = grid.nb_points_y
    x_min, x_max = grid.x_min, grid.x_max
    y_min, y_max = grid.y_min, grid.y_max
    n_steps = len(grid.t)
    ic_fn, kind_used = make_initial_condition_burgers_fn_2d(grid.X, grid.Y, speed, ic_kind)
    for k in range(max_retries):
        try:
            sim = Burgers2D(grid, nu, ic_fn, boundary_condition=boundary_condition, cfl_safety=cfl_safety)
            U = sim.simulate()
            return grid, U, kind_used
        except ValueError as e:
            if "unstable" in str(e).lower():
                dt *= 0.5
                grid = make_grid_2d(nbx, x_min, x_max, nby, y_min, y_max, dt, n_steps=n_steps)
            else:
                raise
        if k == max_retries - 2 and ic_kind is None:
            ic_fn, kind_used = make_initial_condition_burgers_fn_2d(grid.X, grid.Y, speed, None)
    raise RuntimeError("Impossible de stabiliser la simulation après plusieurs essais.")


def generate_dataset_burgers_2d(
    *,
    out_dir="generated_2d_burgers",
    nbx=30, nby=30, x_min=-5, x_max=5, y_min=-5, y_max=5, dt=5e-3, n_steps=20,
    nu=0.1, speed=4.0,
    boundary_condition=bc_periodic_2d,
    ic_kinds=None,
    n_train=100, n_test=20,
    cfl_safety=0.5,
    speed_random=False
):
    """Génération de dataset pour Burger 2D (x, y, t).
    Tous les fichiers sont stockés dans generated_2d_burgers/{train,test}.
    Si speed_random=True et speed est une liste de 2 éléments [min, max], 
    alors les valeurs de speed seront échantillonnées aléatoirement entre min et max.
    """
    train_dir = os.path.join(out_dir, "train")
    test_dir  = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    xmins   = _tolist(x_min)
    xmaxs   = _tolist(x_max)
    ymins   = _tolist(y_min)
    ymaxs   = _tolist(y_max)
    stepsLs = _tolist(n_steps)
    speeds  = _tolist(speed)
    nus     = _tolist(nu)
    print(f"Dataset generation 2D: using nu values: {nus}")
    
    # Vérifier si on doit utiliser des valeurs aléatoires pour speed
    if speed_random and len(speeds) == 2:
        speed_min, speed_max = min(speeds), max(speeds)
        use_random_speed = True
    else:
        use_random_speed = False

    # Alignement des listes x et y
    if len(xmins) != len(xmaxs):
        if len(xmins) == 1:
            xmins = xmins * len(xmaxs)
        elif len(xmaxs) == 1:
            xmaxs = xmaxs * len(xmins)
        else:
            raise ValueError("x_min et x_max doivent avoir la même longueur ou être scalaires.")
    
    if len(ymins) != len(ymaxs):
        if len(ymins) == 1:
            ymins = ymins * len(ymaxs)
        elif len(ymaxs) == 1:
            ymaxs = ymaxs * len(ymins)
        else:
            raise ValueError("y_min et y_max doivent avoir la même longueur ou être scalaires.")
    
    x_pairs = list(zip(xmins, xmaxs))
    y_pairs = list(zip(ymins, ymaxs))

    def do_split(N, split_dir, xmin_val, xmax_val, ymin_val, ymax_val, nst_val, spd_val_or_range, nu_val):
        for i in range(N):
            grid0 = make_grid_2d(nbx, xmin_val, xmax_val, nby, ymin_val, ymax_val, dt, n_steps=int(nst_val))
            kind = None if ic_kinds is None else random.choice(ic_kinds)
            
            # Générer une valeur de speed aléatoire si nécessaire
            if use_random_speed:
                actual_speed = random.uniform(speed_min, speed_max)
            else:
                actual_speed = float(spd_val_or_range)
            
            grid_i, U, kind_used = run_one_sim_burgers_2d(
                grid0, nu=float(nu_val), speed=actual_speed,
                ic_kind=kind, cfl_safety=cfl_safety, boundary_condition=boundary_condition
            )

            tag = f"{kind_used}|x=[{xmin_val},{xmax_val}]|y=[{ymin_val},{ymax_val}]|T={int(nst_val)}|v={actual_speed:.3f}"
            filename = (
                f"sample_{i:04d}"
                f"_x{float(xmin_val):+.2f}_{float(xmax_val):+.2f}"
                f"_y{float(ymin_val):+.2f}_{float(ymax_val):+.2f}"
                f"_v{actual_speed:.3f}"
                f"_T{int(nst_val)}.npz"
            )

            grid_i.save_npz(
                os.path.join(split_dir, filename),
                U, nu=float(nu_val), speed=actual_speed, tag=tag
            )

    # Adaptation selon le mode de génération de speed
    if use_random_speed:
        # Si on utilise des valeurs aléatoires, on ne fait qu'une seule itération
        for (xmin, xmax) in x_pairs:
            for (ymin, ymax) in y_pairs:
                for nst in stepsLs:
                    for nu_v in nus:
                        do_split(n_train, train_dir, xmin, xmax, ymin, ymax, nst, None, nu_v)
                        do_split(n_test,  test_dir,  xmin, xmax, ymin, ymax, nst, None, nu_v)
    else:
        # Mode normal : itérer sur chaque valeur de speed
        for (xmin, xmax) in x_pairs:
            for (ymin, ymax) in y_pairs:
                for nst in stepsLs:
                    for spd in speeds:
                        for nu_v in nus:
                            do_split(n_train, train_dir, xmin, xmax, ymin, ymax, nst, spd, nu_v)
                            do_split(n_test,  test_dir,  xmin, xmax, ymin, ymax, nst, spd, nu_v)


def visualize_random_sample_2d(
    out_dir="generated_2d_burgers/test",
    cmap="seismic",
    label="U",
    title_prefix="Burgers 2D",
    show_animation=True,
    interval_ms=60,
    save_video_path=None
):
    """Visualise un échantillon aléatoire du dataset 2D.
    - show_animation=True affiche l'animation.
    - save_video_path peut être .mp4 ou .gif pour sauvegarder la vidéo.
    """
    files = [f for f in os.listdir(out_dir) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"Aucun .npz dans {out_dir}")
    sample = random.choice(files)
    path = os.path.join(out_dir, sample)
    print(f"📂 Loaded file: {sample}")

    data = np.load(path, allow_pickle=True)
    U = data["U"]; x = data["x"]; y = data["y"]; t = data["t"]; dt = float(data["dt"])
    tag = data["tag"].item() if "tag" in data else ""
    nu  = float(data["nu"]) if "nu" in data.files else None
    spd = float(data["speed"]) if "speed" in data.files else None

    try:
        from visualization import show_field
    except Exception:
        project_root = Path(__file__).parent.parent
        ml_dev = project_root / "ML_Development"
        if str(ml_dev) not in sys.path:
            sys.path.insert(0, str(ml_dev))
        from visualization import show_field

    show_field(
        U,
        title_prefix=title_prefix,
        cmap=cmap,
        interval_ms=interval_ms,
        show_animation=show_animation,
        save_video_path=save_video_path,
        x=x,
        y=y,
        t=t,
        label=label,
    )


if __name__ == "__main__":
    # Your data loading
    project_root = Path(__file__).parent.parent
    dir = project_root / "saved_dataset/generated_2d_burgers"
    # dir = "test/"

    print("start generating 2D Burgers dataset")
    generate_dataset_burgers_2d(
        out_dir= dir,
        nbx=64,  # Résolution en x
        nby=64,  # Résolution en y
        x_min=[-5],
        x_max=[5],
        y_min=[-5],
        y_max=[5],
        dt=5e-2,
        n_steps=[128],  # Pas de temps réduits pour la stabilité en 2D
        # nu=np.linspace(0.01, 0.5, 10),  # Moins de valeurs pour tester
        nu=[0.1],
        speed=[1.0, 5.0],
        speed_random=True,  # Active la génération aléatoire entre 1.0 et 5.0
        boundary_condition=bc_neumann_zero_2d,
        ic_kinds=["shock","rarefaction","sine","smooth","hyperbolic_tangent","radial"],
        n_train=100, n_test=30,  # Nombres réduits pour tester
        cfl_safety=0.8
    )

    print("generation end")
    visualize_random_sample_2d(os.path.join(dir, "train"))