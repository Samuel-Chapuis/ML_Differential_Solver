import numpy as np

class Grid1D:
    def __init__(self, nb_points_x, x_min, x_max, t_max, dt):
        self.nb_points_x = nb_points_x
        self.x_min, self.x_max = x_min, x_max
        self.dx = (x_max - x_min) / (nb_points_x - 1)
        self.x = np.linspace(x_min, x_max, nb_points_x)
        self.t_max, self.dt = t_max, dt
        self.nb_points_t = int(round(t_max / dt)) + 1
        self.t = np.linspace(0.0, t_max, self.nb_points_t)

    def save_npz(self, filename, U, nu=None, speed=None, tag=None):
        # Keep x and t so the dataloader can recover dx, dt precisely.
        np.savez(filename, U=U, nu=nu, speed=speed, tag=tag, x=self.x, t=self.t)

def _periodic_index(arr):
    """Convenience for periodic neighbors."""
    return np.roll(arr, -1), np.roll(arr, 1)

def sample_initial_condition_burgers(grid, nu, speed, ic_kind="fourier", Kmax=4, seed=None):
    """
    Periodic ICs compatible with periodic BCs.
    - 'sine'     : single sine with integer wavenumber
    - 'two_mode' : sum of two sines (integer modes)
    - 'fourier'  : random Fourier (1..Kmax) with random phases (default)
    """
    rng = np.random.default_rng(seed)
    x = grid.x
    L = grid.x_max - grid.x_min
    xi = 2.0 * np.pi * (x - grid.x_min) / L

    if ic_kind == "sine":
        k = rng.integers(1, 5)  # integer mode
        u0 = speed * np.sin(k * xi)
    elif ic_kind == "two_mode":
        k1 = rng.integers(1, 4)
        k2 = rng.integers(1, 4)
        a1, a2 = rng.uniform(0.4, 1.0, size=2)
        phi1, phi2 = rng.uniform(0, 2*np.pi, size=2)
        u0 = speed * (a1*np.sin(k1*xi + phi1) + a2*np.sin(k2*xi + phi2))
    else:  # 'fourier' (default)
        u0 = np.zeros_like(x)
        for k in range(1, Kmax+1):
            a_k = rng.normal(0, 1.0) / k
            b_k = rng.normal(0, 1.0) / k
            u0 += a_k * np.sin(k*xi) + b_k * np.cos(k*xi)
        u0 = speed * u0 / (np.max(np.abs(u0)) + 1e-12)
    return u0

def solve_burgers_conservative(grid, nu, u0, cfl_safety=0.4):
    """
    Viscous Burgers: u_t + (u^2/2)_x = nu * u_xx
    Conservative advection (Rusanov) + explicit diffusion.
    Periodic BCs.
    """
    nx = grid.nb_points_x
    dx = grid.dx
    t_max = grid.t_max

    # Internal stable dt: min(CFL, diffusion, requested output dt)
    umax = max(1e-8, np.max(np.abs(u0)))
    dt_cfl  = cfl_safety * dx / umax
    dt_diff = cfl_safety * dx*dx / max(nu, 1e-12)
    dt_int  = min(grid.dt, dt_cfl, dt_diff)

    nt_int = int(np.ceil(t_max / dt_int)) + 1
    dt_int = t_max / (nt_int - 1)  # hit t_max exactly

    U_int = np.zeros((nt_int, nx))
    U_int[0] = u0.copy()
    t_int = np.linspace(0.0, t_max, nt_int)

    for n in range(nt_int - 1):
        u = U_int[n]

        # periodic neighbors
        u_right = np.roll(u, -1)
        u_left  = np.roll(u,  1)

        # Rusanov flux for f(u)=u^2/2 at interfaces i+1/2
        # F_{i+1/2} = 0.5*(f(u_i)+f(u_{i+1})) - 0.5*alpha*(u_{i+1}-u_i)
        f = 0.5 * u**2
        f_right = 0.5 * u_right**2
        alpha = np.maximum(np.abs(u), np.abs(u_right))
        F_iphalf = 0.5*(f + f_right) - 0.5*alpha*(u_right - u)

        # flux at i-1/2 (just shift)
        f_left = 0.5 * u_left**2
        alpha_l = np.maximum(np.abs(u_left), np.abs(u))
        F_imhalf = 0.5*(f_left + f) - 0.5*alpha_l*(u - u_left)

        adv_update = -(F_iphalf - F_imhalf) / dx

        # diffusion (second central difference)
        u_xx = (u_right - 2*u + u_left) / (dx*dx)
        diff_update = nu * u_xx

        u_next = u + dt_int * (adv_update + diff_update)

        # simple safety clamp (rarely triggered if dt_int chosen well)
        u_next = np.clip(u_next, -100.0, 100.0)
        U_int[n+1] = u_next

    # --- Linear resampling in time to match the requested output grid.times ---
    if len(grid.t) != len(t_int):
        t_out = grid.t
        U_out = np.empty((len(t_out), nx))
        for i in range(nx):
            U_out[:, i] = np.interp(t_out, t_int, U_int[:, i])
    else:
        U_out = U_int
    return U_out

def run_one_sim_burgers(grid, nu, speed=1.0, ic_kind="fourier"):
    u0 = sample_initial_condition_burgers(grid, nu, speed, ic_kind=ic_kind)
    U = solve_burgers_conservative(grid, nu, u0)
    return grid, U, ic_kind

# Compatibility placeholder
class Burgers1D:
    pass

if __name__ == "__main__":
    # quick sanity plot
    import matplotlib.pyplot as plt
    grid = Grid1D(nb_points_x=141, x_min=-5, x_max=5, t_max=1.0, dt=0.01)
    for nu, speed, ic in [(0.01, 1.5, "fourier"), (0.1, 1.0, "sine")]:
        g, U, tag = run_one_sim_burgers(grid, nu, speed, ic)
        plt.figure(figsize=(6,3))
        plt.imshow(U, aspect='auto', origin='lower',
                   extent=[g.x[0], g.x[-1], 0, g.t_max], cmap='viridis')
        plt.title(f"nu={nu}, ic={tag}")
        plt.xlabel("x"); plt.ylabel("t")
        plt.tight_layout()
        plt.show()