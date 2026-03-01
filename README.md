# ML Differential Solver — NFTM for Burgers Dynamics

**Big Data Research Project — Big Data Management & Analytics**
CentraleSupelec & Universite Paris-Saclay

---

## Abstract

Solving partial differential equations (PDEs) efficiently remains a central challenge in computational physics. Traditional numerical methods produce accurate results but scale poorly with problem complexity, motivating data-driven neural surrogates that learn solution operators directly from simulation data. This work investigates the Neural Field Turing Machine (NFTM) framework for PDE surrogate modelling, in which a neural controller reads from and writes to a continuous external spatial memory to advance the system state in time. We develop and evaluate three controller architectures — CNN, Transformer, and a Causal Temporal Convolutional Attention Network (TCAN) — as autoregressive simulators of the 1D viscous Burgers equation. TCAN, which combines spatial 1D convolutions with a causal temporal attention mechanism and explicit viscosity conditioning via Feature-wise Linear Modulation (FiLM), consistently outperforms the other controllers on long-horizon rollouts. Trained and evaluated on the official FNO benchmark dataset at four spatial resolutions (s in {85, 141, 211, 421}), TCAN achieves a relative L2 error of 0.30 at s=85, compared to FNO's 0.011. The gap is attributed to autoregressive error accumulation over up to 256 rollout steps. The TCAN architecture is further extended to 2D Burgers dynamics.

**Keywords:** Neural Field Turing Machines, Partial Differential Equations, Burgers Equation, Operator Learning, Fourier Neural Operator, Temporal Convolutional Networks, Causal Attention, Autoregressive Surrogates, Viscosity Conditioning, FiLM Modulation, Progressive Rollout Training, Scientific Machine Learning, Computational Fluid Dynamics.

---

## Team

**Students:**
- Samuel Chapuis — samuel.chapuis@student-cs.fr
- Lucia Victoria Fernandez Sanchez — lucia-victoria.fernandez@student-cs.fr
- Alexandra Perruchot-Triboulet Rodriguez — alexandra.perruchot-triboulet-rodriguez@student-cs.fr

**Supervisors:**
- Nacera Seghouani — Nacera.Seghouani@centralesupelec.fr
- Akash Malhotra — akash.malhotra@centralesupelec.fr

---

## Results

| Method | s=85 | s=141 | s=211 | s=421 |
|--------|------|-------|-------|-------|
| FNO (Li et al., 2021) | **0.011** | **0.011** | **0.011** | **0.010** |
| TCAN (ours) | 0.303 | 0.335 | 0.726 | 0.944 |

Metric: relative L2 error. Lower is better. FNO results taken from Li et al. (2021).

The performance gap is explained by the autoregressive formulation: TCAN accumulates small per-step errors over up to 256 rollout steps, whereas FNO maps the initial condition directly to the terminal solution in a single forward pass.

---

## Repository Structure

```
ML_Differential_Solver/
│
├── ML_Development/                  # Core ML code
│   ├── models.py                    # CNN, Transformer, TCAN model definitions
│   ├── training.py                  # Training loops and curriculum learning
│   ├── data_loading.py              # FNO benchmark dataset loader
│   ├── evaluation.py                # Metrics: MSE, SSIM, PSNR, PDE residual
│   ├── visualization.py             # Trajectory heatmaps and error plots
│   ├── main.ipynb                   # Main 1D Burgers experiment notebook
│   ├── main_85x85.ipynb             # Experiment at s=85 resolution
│   ├── main2d.ipynb                 # 2D Burgers extension
│   └── Proof_Nu05.ipynb             # Viscosity conditioning experiment
│
├── Physical_Simulator/              # Reference numerical PDE solvers
│   ├── burger1D.py                  # 1D Burgers pseudo-spectral solver
│   ├── burger2D.py                  # 2D Burgers solver
│   └── burgers_minimal_*.py         # Minimal reference implementations
│
├── saved_models/
│   └── TCAN_100epochs_Best/         # Best trained TCAN weights (1D, s=85)
│
├── saved_results/
│   └── new_dataset_results/TCAN/    # Evaluation outputs at all resolutions
│
├── samples/                         # Sample trajectory visualizations
│   ├── sample_85x85_nu0001.png      # Low-viscosity (sharp shock)
│   ├── sample_85x85_nu03.png        # High-viscosity (smooth)
│   └── ...                          # 4 resolutions x 2 viscosity values
│
├── video/                           # Animated GIF predictions
│   ├── train_sample.gif
│   ├── test_sample.gif
│   └── prediction_nu_*.gif
│
├── Documentation/
│   └── Final version/
│       ├── Report/                  # LaTeX thesis source
│       │   └── build/Thesis.pdf     # Compiled thesis (57 pages)
│       └── Presentation/            # Beamer slides source
│           └── build/main.pdf       # Compiled presentation
│
└── deprecated/                      # Legacy CNN/RNN experiments
```

---

## Setup

### Requirements

- Python 3.9+
- PyTorch >= 2.0
- NumPy, SciPy
- Matplotlib
- scikit-image (for SSIM/PSNR)
- Jupyter

```bash
pip install torch numpy scipy matplotlib scikit-image jupyter
```

### Dataset

This project uses the official FNO benchmark dataset for the 1D viscous Burgers equation.
Download the `.mat` files from the FNO repository and place them in a `data/` folder:

```
https://github.com/neuraloperator/neuraloperator
```

The data loader in `ML_Development/data_loading.py` handles downsampling to each target resolution.

---

## Running Experiments

### 1D Burgers — Main Experiment

Open `ML_Development/main.ipynb`. The notebook covers:
1. Dataset loading and visualization at all 4 resolutions
2. Training CNN, Transformer, and TCAN controllers
3. Autoregressive rollout evaluation
4. Comparison table vs FNO

### 2D Burgers Extension

Open `ML_Development/main2d.ipynb`.

### Training TCAN from scratch

```python
from models import TCANController
from training import train_tcan

model = TCANController(window_size=20, channels=32, nu_conditioning=True)
train_tcan(model, train_loader, epochs=100,
           rollout_depth_schedule=[8, 16, 64])
```

---

## Model Architecture — TCAN

TCAN predicts the next field $f_{t+1}$ from a sliding window of the last $W=20$ fields:

```
Input: history window  (B, W, N)  +  viscosity nu
   |
   v
Frame-wise Conv1d + GELU         ->  features  (B, W, 32, N)
   |
   +-- Query (last frame)
   +-- Keys, Values (all frames)
   |
   v
Causal Temporal Attention        ->  context   (B, 32, N)
   |
   v
FiLM viscosity conditioning      ->  conditioned context
   |
   v
Decoder Conv1d                   ->  correction field
   |
   v
tanh(.) x 0.1                    ->  bounded correction  |delta f| <= 0.1
   |
   v
Residual: f_{t+1} = f_t + delta f
```

---

## Training Strategy

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW with cosine annealing |
| Epochs | 100 |
| Progressive rollout depth | 8 -> 16 -> 64 steps |
| Loss | MSE + gradient penalty + energy dissipation |
| Teacher forcing | Gradual transition to model predictions |
| Batch size | 16 |

---

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Relative L2 error | Primary FNO benchmark metric |
| MSE | Pixel-level accuracy |
| SSIM / PSNR | Structural and perceptual quality |
| Mass conservation error | Physics consistency |
| Energy monotonicity | Dissipation correctness |
| PDE residual | Direct PDE satisfaction |
| Gradient error | Shock sharpness |

---

## Limitations and Next Steps

**Known limitations:**
- Autoregressive error accumulation over 256 steps is the primary performance bottleneck vs FNO
- Symmetric (reflect) padding violates periodic boundary conditions; circular padding would fix this
- 2D model trains stably but requires more data for quantitative accuracy

**Planned improvements:**
- Replace reflect padding with circular padding across all convolutional layers
- Scale 2D dataset to 1000 train / 200 test trajectories per resolution
- Extend to 2D incompressible Navier-Stokes with Hodge-projection pressure correction
- Apply PDE-Refiner post-processing for improved shock sharpness

---

## Compiling the Documentation

```bash
# Report
cd "Documentation/Final version/Report"
pdflatex -output-directory=build Thesis.tex
cd build && BIBINPUTS="../:" bibtex Thesis && cd ..
pdflatex -output-directory=build Thesis.tex
pdflatex -output-directory=build Thesis.tex

# Presentation
cd "Documentation/Final version/Presentation"
pdflatex -output-directory=build main.tex
cp build/main.toc . && pdflatex -output-directory=build main.tex && rm main.toc
```

---

## References

1. Li, Z. et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021.
2. Malhotra, A. et al. (2025). *Neural Field Turing Machines*.
3. Kovachki, N. et al. (2023). *Neural Operator: Learning Maps Between Function Spaces*. JMLR 2023.
4. Lippe, P. et al. (2023). *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers*. NeurIPS 2023.
5. Huang, Z. et al. (2025). *PhysicsCorrect: Training-Free Physics Correction for Neural PDE Solvers*.
6. Musekamp, D. et al. (2025). *Active Learning for Neural PDE Solvers*.

---

**Last updated:** March 2026
**Status:** Final submission
