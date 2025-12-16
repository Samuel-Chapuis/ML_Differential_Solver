# Partial Differential Equation Solver using Neural Field Turing Machines (NFTM)

**Master's Thesis - Big Data Management & Analytics**  
CentraleSupélec & Université Paris-Saclay

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

### Immediate (1-2 weeks)
- [x] Fix boundary padding (symmetric → replication)
- [x] Optimize training performance
- [ ] Verify viscosity generalization
- [ ] Set up Weights & Biases for hyperparameter tuning

### Short-term (1-2 months)
- [ ] Implement teacher forcing
- [ ] **Extend to 2D Burgers equation**
- [ ] Compare with Neural ODE and PINN baselines
- [ ] Systematic hyperparameter optimization

### Medium-term (3-4 months)
- [ ] **Incompressible Navier-Stokes** implementation
- [ ] Multi-PDE generalization (Heat equation, Reaction-Diffusion)
- [ ] Equation discovery from data

### Long-term (ongoing)
- [ ] Conservation law enforcement research
- [ ] Long-horizon prediction stability
- [ ] Publication preparation

---

## 📚 Key References

1. **Neural Turing Machines** - Graves et al. (2014)
2. **Physics-Informed Neural Networks** - Raissi et al. (2019)
3. **Fourier Neural Operator** - Li et al. (2021)
4. **Neural Operators** - Kovachki et al. (2023)
5. **Spectral Bias in Neural Networks** - Rahaman et al. (2019)
6. **Neural Field Turing Machine** - Malhotra & Seghouani (2025)

---



## 📝 License & Citation

If you use this work, please cite:

```bibtex
@mastersthesis{chapuis2025nftm,
  title={Partial Differential Equation Solver using Neural Field Turing Machines},
  author={Chapuis, Samuel and Fernandez Sanchez, Lucia Victoria and Perruchot-Triboulet Rodriguez, Alexandra},
  school={CentraleSupélec, Université Paris-Saclay},
  year={2025},
  type={Master's Thesis}
}
```

---

**Last Updated**: December 2024  
**Status**: Active Development 🚧
