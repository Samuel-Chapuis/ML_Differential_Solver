# Partial Differential Equation Solver using Neural Field Turing Machines (NFTM)

**Big Data Research Project - Big Data Management & Analytics**  
CentraleSupélec & Université Paris-Saclay

---

## 📄 Abstract

Solving partial differential equations (PDEs) efficiently remains a fundamental challenge in computational physics, with traditional numerical methods often requiring significant computational resources for complex, multi-scale dynamics. This work presents a Neural Field Turing Machine (NFTM) approach for learning PDE solutions, combining external continuous spatial memory with learned read/write operations through a neural controller. We develop architectures as data-driven simulators for the viscous Burgers equation, a canonical nonlinear PDE that captures essential features of fluid dynamics including shock formation and nonlinear advection. Our approach employs causal temporal attention to aggregate information across history windows, bounded residual corrections for numerical stability during long autoregressive rollouts, and physics-informed loss terms enforcing energy dissipation and gradient accuracy. Trained on 724 trajectories spanning viscosity values from 0.001 to 0.5, the models are evaluated on both standard metrics (MSE, SSIM, PSNR) and physics-informed criteria (mass conservation, energy monotonicity, PDE residual). This framework establishes a foundation for generalizable neural PDE solvers, with planned extensions to two-dimensional Burgers and incompressible Navier-Stokes equations.

**Keywords:** Neural Field Turing Machines, Physics-Informed Neural Networks, Partial Differential Equations, Burgers Equation, Temporal Convolutional Networks, Attention Mechanisms, Autoregressive Models, Scientific Machine Learning, Computational Fluid Dynamics, Deep Learning for Physics.

---

## 👥 Team

**Students:**
- Samuel CHAPUIS - samuel.chapuis@student-cs.fr
- Lucia Victoria FERNANDEZ SANCHEZ - lucia-victoria.fernandez@student-cs.fr
- Alexandra PERRUCHOT-TRIBOULET RODRIGUEZ - alexandra.perruchot-triboulet-rodriguez@student-cs.fr

**Advisors:**
- Nacéra SEGHOUANI - Nacera.Seghouani@centralesupelec.fr
- Akash MALHOTRA - akash.malhotra@centralesupelec.fr

---

## 🎯 Project Overview

This project develops a **generic neural architecture** for solving partial differential equations (PDEs), with a focus on fluid mechanics simulations. The goal is to build a model that can internalize governing physical laws and accurately simulate complex, multi-scale, nonlinear dynamics across different PDE families.

### Key Objectives
-  Learn time-evolution operators across multiple PDE families
-  Simulate and extrapolate trajectories beyond training horizons
-  Maintain stability and accuracy during long-term predictions
-  Generalize across different physical parameters (e.g., viscosity)

---

## 🔬 Current Focus: 1D Burgers Equation

We begin with the **viscous Burgers equation**, a simplified 1D version of Burgers equation that captures key fluid dynamics features:

```
∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²)
```

**Why Burgers equation?**
- Simple enough for limited data/computational resources
- Rich dynamics: shock formation, nonlinear advection
- Benchmark for testing neural PDE solvers
- Natural stepping stone to full Navier-Stokes

---



## 🎓 Methodology

### Training Procedure
1. **Curriculum Learning**: Progressive rollout depth (8 → 16 → 64 steps)
2. **Composite Loss Function**:
   - MSE (accuracy)
   - Gradient matching (∇ₓu)
   - Energy dissipation constraint
3. **Teacher Forcing**: Gradual transition from ground truth to predictions
4. **Optimizer**: AdamW with cosine annealing

### Evaluation Metrics

**Accuracy Metrics:**
- Mean Squared Error (MSE)
- Relative L² error
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

**Physics-Informed Metrics:**
- Mass conservation error
- Energy monotonicity
- PDE residual
- Spectral error
- Maximum gradient error

---

## 🚀 Next Steps


- Merge architectures to have a generalized model
- Verify viscosity generalization (no sending the viscosity as parameter)
- Set up Weights & Biases for hyperparameter tuning
- Systematic hyperparameter optimization
- **Extend to 2D Burgers equation** (first try without articles, then add objects inside flow field)
- **Incompressible Navier-Stokes** implementation (include pressure)
- Long-horizon prediction stability
- Publication preparation


---

## 📚 Key References

1. **Neural Turing Machines** - Graves et al. (2014)
2. **Physics-Informed Neural Networks** - Raissi et al. (2019)
3. **Fourier Neural Operator** - Li et al. (2021)
4. **Neural Operators** - Kovachki et al. (2023)
5. **Spectral Bias in Neural Networks** - Rahaman et al. (2019)
6. **Neural Field Turing Machine** - Malhotra & Seghouani (2025)

---

**Last Updated**: December 2025
**Status**: Working Progress