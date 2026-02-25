# import numpy as np

# class Grid1D:
#     def __init__(self, nb_points_x, x_min, x_max, t_max, dt):
#         self.nb_points_x = nb_points_x
#         self.x_min, self.x_max = x_min, x_max
#         self.dx = (x_max - x_min) / (nb_points_x - 1)
#         self.x = np.linspace(x_min, x_max, nb_points_x)
#         self.t_max, self.dt = t_max, dt
#         self.nb_points_t = int(t_max / dt) + 1
#         self.t = np.linspace(0, t_max, self.nb_points_t)
    
#     def save_npz(self, filename, U, nu=None, speed=None, tag=None):
#         np.savez(filename, U=U, nu=nu, speed=speed, tag=tag, x=self.x, t=self.t)

# def sample_initial_condition_burgers(grid, nu, speed, ic_kind):
#     x = grid.x
#     if ic_kind == "shock":
#         u0 = (x > 0).astype(float) * speed
#     elif ic_kind == "sine":
#         u0 = speed * np.sin(np.pi * x)
#     elif ic_kind == "rarefaction":
#         u0 = -speed * np.tanh(x)
#     elif ic_kind == "smooth":
#         u0 = speed * np.exp(-x**2)
#     return u0

# def solve_burgers_fast(grid, nu, u0):
#     """
#     Fast explicit solver for Burgers equation using upwind scheme.
#     Optimized with vectorized operations and automatic stable time stepping.
    
#     This is much faster than implicit methods while maintaining stability.
#     """
#     nx = grid.nb_points_x
#     dx = grid.dx
#     t_max = grid.t_max
    
#     # Automatic stable time step
#     max_u = np.max(np.abs(u0))
#     if max_u > 1e-10:
#         # CFL condition: dt < dx / max|u|
#         dt_cfl = 0.4 * dx / max_u  # Use 0.4 for safety margin
#     else:
#         dt_cfl = 0.01
    
#     # Diffusion stability: dt < dx^2 / (2*nu)
#     dt_diff = 0.4 * dx**2 / (nu + 1e-10)
    
#     # Take the minimum
#     dt = min(dt_cfl, dt_diff, grid.dt)
    
#     # Number of time steps needed
#     nt_actual = int(np.ceil(t_max / dt)) + 1
#     dt = t_max / (nt_actual - 1)  # Adjust to hit t_max exactly
    
#     # Allocate solution array
#     U = np.zeros((nt_actual, nx))
#     U[0] = u0.copy()
    
#     # Precompute constants for efficiency
#     dtdx = dt / dx
#     dtdx2 = dt / (dx * dx)
#     nu_dtdx2 = nu * dtdx2
    
#     # Time stepping loop
#     for n in range(nt_actual - 1):
#         u = U[n]
        
#         # VECTORIZED upwind scheme for advection
#         # For u > 0: use backward difference
#         # For u < 0: use forward difference
#         u_left = np.roll(u, 1)   # u[i-1] with periodic BC
#         u_right = np.roll(u, -1)  # u[i+1] with periodic BC
        
#         # Upwind selection
#         du_backward = u - u_left
#         du_forward = u_right - u
        
#         # Where u > 0, use backward; where u < 0, use forward
#         du = np.where(u > 0, du_backward, du_forward)
        
#         # Advection term: -u * du/dx
#         advection = -u * du / dx
        
#         # VECTORIZED diffusion term: nu * d2u/dx2
#         d2u = u_right - 2*u + u_left
#         diffusion = nu_dtdx2 * d2u
        
#         # Update
#         U[n+1] = u + dt * advection + diffusion
        
#         # Safety check for numerical stability (optional, can comment out for speed)
#         if np.any(np.abs(U[n+1]) > 1e5):
#             print(f"Warning: Large values detected at step {n}, limiting...")
#             U[n+1] = np.clip(U[n+1], -100, 100)
    
#     # Subsample to match desired output resolution
#     if nt_actual != grid.nb_points_t:
#         t_indices = np.linspace(0, nt_actual-1, grid.nb_points_t, dtype=int)
#         U = U[t_indices]
    
#     return U

# def run_one_sim_burgers(grid, nu, speed=1.0, ic_kind="shock", method="fast"):
#     """
#     Run one Burgers equation simulation
    
#     Parameters
#     ----------
#     grid : Grid1D
#         Spatial and temporal grid
#     nu : float
#         Viscosity coefficient
#     speed : float
#         Amplitude/speed parameter for initial condition
#     ic_kind : str
#         Type of initial condition: "shock", "sine", "rarefaction", "smooth"
#     method : str
#         Numerical method (default "fast")
    
#     Returns
#     -------
#     grid : Grid1D
#         The grid object
#     U : ndarray
#         Solution array of shape (nt, nx)
#     ic_kind : str
#         The initial condition type used
#     """
#     # Generate initial condition
#     u0 = sample_initial_condition_burgers(grid, nu, speed, ic_kind)
    
#     # Solve
#     U = solve_burgers_fast(grid, nu, u0)
    
#     return grid, U, ic_kind

# # Burgers1D class (placeholder for compatibility)
# class Burgers1D:
#     pass


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import time
    
#     # Speed test
#     print("Speed test: Generating 10 samples at 141x141 resolution...")
#     grid = Grid1D(nb_points_x=141, x_min=-5, x_max=5, t_max=1.4, dt=0.01)
    
#     start = time.time()
#     for i in range(10):
#         nu = np.random.choice([0.001, 0.01, 0.1, 0.5])
#         speed = np.random.uniform(0.5, 2.0)
#         ic = np.random.choice(["shock", "sine"])
#         grid_out, U, kind = run_one_sim_burgers(grid, nu, speed=speed, ic_kind=ic)
    
#     elapsed = time.time() - start
#     print(f"✓ 10 samples generated in {elapsed:.2f} seconds ({elapsed/10:.3f} sec/sample)")
#     print(f"  Estimated time for 500 samples: {elapsed*50:.1f} seconds = {elapsed*50/60:.1f} minutes")
    
#     # Visual test
#     print("\nVisual test...")
#     test_cases = [
#         (0.01, 1.5, "shock"),
#         (0.7, 1.5, "shock"),
#         (0.009, 2.0, "sine"),
#     ]
    
#     fig, axes = plt.subplots(len(test_cases), 2, figsize=(14, 4*len(test_cases)))
    
#     for idx, (nu, speed, ic) in enumerate(test_cases):
#         print(f"  Testing nu={nu}, speed={speed}, ic={ic}...")
#         grid_out, U, kind = run_one_sim_burgers(grid, nu, speed=speed, ic_kind=ic)
#         print(f"    ✓ Shape: {U.shape}, max value: {np.max(np.abs(U)):.3f}")
        
#         # Plot profile
#         axes[idx, 0].plot(grid_out.x, U[0], 'b-', label='t=0')
#         axes[idx, 0].plot(grid_out.x, U[len(U)//2], 'r-', label=f't={grid_out.t_max/2:.2f}')
#         axes[idx, 0].plot(grid_out.x, U[-1], 'g-', label=f't={grid_out.t_max:.2f}')
#         axes[idx, 0].set_title(f'nu={nu}, speed={speed}, {ic}')
#         axes[idx, 0].set_xlabel('x')
#         axes[idx, 0].set_ylabel('u')
#         axes[idx, 0].legend()
#         axes[idx, 0].grid(True)
        
#         # Plot heatmap
#         im = axes[idx, 1].imshow(U, aspect='auto', origin='lower', cmap='viridis',
#                                   extent=[grid_out.x[0], grid_out.x[-1], 0, grid_out.t_max])
#         axes[idx, 1].set_title('Space-Time Evolution')
#         axes[idx, 1].set_xlabel('x')
#         axes[idx, 1].set_ylabel('t')
#         plt.colorbar(im, ax=axes[idx, 1])
    
#     plt.tight_layout()
#     plt.savefig('burgers_fast_test.png', dpi=150, bbox_inches='tight')
#     print("\n✓ Test plot saved as burgers_fast_test.png")

# burger.py  (conservative + periodic ICs)
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