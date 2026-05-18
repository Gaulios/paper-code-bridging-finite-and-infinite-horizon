# Code for "Bridging Finite and Infinite-Horizon Nash Equilibria in Linear Quadratic Games"

This repository contains the Python code used to generate the figures in:

> G. Salizzoni, S. Hall, and M. Kamgarpour,  
> **"Bridging Finite and Infinite-Horizon Nash Equilibria in Linear Quadratic Games"**,  
> IEEE Conference on Decision and Control (CDC), 2026.

---

## Repository structure

```
.
├── utils.py                        # Core functions: Riccati recursion, game generation, plotting
│
├── Fig1_create_and_plot.py         # Generate data and produce Figure 1 (convergence heatmaps)
├── Fig1_just_plot.py               # Reproduce Figure 1 from saved data
│
├── Fig2_create_and_plot.py         # Generate data and produce Figure 2 (scalar convergence map)
├── Fig2_just_plot.py               # Reproduce Figure 2 from saved data
│
├── Fig3_create_and_plot.py         # Generate data and produce Figure 3 (periodic NE example)
├── Fig3_just_plot.py               # Reproduce Figure 3 from saved data
│
├── FigExtra_create_and_plot.py     # Generate data and produce policy-evolution plot (not in paper)
├── FigExtra_just_plot.py           # Reproduce policy-evolution plot from saved data
│
├── data/                           # Pre-generated datasets used in the paper
│   ├── Fig1_data_paper.pkl
│   ├── Fig2_data_paper.pkl
│   ├── Fig3_data_paper.pkl
│   └── FigExtra_data_paper.pkl
│
└── figures/                        # Output directory for generated figures (created on first run)
```

---

## Quick start: reproduce the paper figures

```bash
python Fig1_just_plot.py   # Figure 1 — convergence heatmaps
python Fig2_just_plot.py   # Figure 2 — scalar convergence map
python Fig3_just_plot.py   # Figure 3 — periodic Nash equilibrium example
```

Figures are saved as PDF files in the `figures/` directory.

---

## Regenerating the data from scratch

Each `*_create_and_plot.py` script resamples the data and overwrites the corresponding
file in `data/`.  Note that Fig2 requires generating 10^4 random games for each
(n, N) pair and may take several hours.

```bash
python Fig1_create_and_plot.py
python Fig2_create_and_plot.py
python Fig3_create_and_plot.py
```

---

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

Install with:
```bash
pip install numpy scipy matplotlib
```
