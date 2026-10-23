# Replication package — The Discarded Spectrum

- **Repository:** https://github.com/sun0720/002_low-rank
- **DOI:** [CONFIRM] archive DOI once the release is minted
- **License:** MIT — see `LICENSE`
- **Environment:** Python 3.11; `requirements.txt` is authoritative where the description below is less specific

## Contents

- **Scripts (entry points):** `approx_cell.py`, `joint_cell.py`, `joint_grid.py`, `poly_cell.py`, `sim.py`
- **Outputs:** `results/`

## Package description

This package reproduces the simulation evidence in Section 5 and Tables 2–7 of the manuscript. It contains only synthetic data. No individual-level or proprietary data are used.

## Environment

- Python 3.11
- NumPy 2.x
- Matplotlib (only for the figure files)

The manuscript's reported runs used Python 3.11 with NumPy 2.x.

## Run

From this directory:

```bash
python3 sim.py          # main scalar grid, spectral grid, heteroskedastic design
python3 poly_cell.py    # zero-bias polynomial control
python3 joint_cell.py   # joint certified design (nonzero bias, bounded Y)
python3 joint_grid.py   # critical-sequence grid, delta_n = c n^{-1/4}
python3 approx_cell.py  # rate-compliant and small-overlap cells
```

Each script writes its CSV to `results/` and prints the cell-level summary. The master seeds are in the scripts (`20261020` for the main grid, `20261021` for the critical-sequence grid, `20261022` for the approximation/small-overlap cells).

## Mapping to the manuscript

| Manuscript object | Script | Output |
|---|---|---|
| Tables 2–3, Figures 1–6 | `sim.py` | `results/results.csv`, `results/summary.txt` |
| Table 4, Figure 7 (rank profile) | `sim.py` (spectral design) | `results/results.csv` |
| Zero-bias polynomial control | `poly_cell.py` | `results/results_poly.csv` |
| Joint certified design | `joint_cell.py` | `results/results_joint.csv` |
| Critical-sequence grid | `joint_grid.py` | `results/results_joint_grid.csv` |
| Rate-compliant and small-overlap cells | `approx_cell.py` | `results/results_approx.csv` |
| P9 algebraic check | `sim.py` | `results/results_p9.csv` |

## Notes

- The main scalar grid uses a growing sieve degree `k_n = max{4, 2 + floor(log10 n)}` and is a descriptive stress design outside A7(iv); the fixed-degree robustness cells use degree 6. This distinction is stated in Section 5.1.
- The rate-compliant cell in `approx_cell.py` is retained as a descriptive approximation-rate check. Under the reported joint-design constants it is not certified, because the degree-5 component also changes the pathwise gradient and second-derivative bounds; see the manuscript's Section 5.1 note.
- `results/*.csv` are the exact outputs underlying the reported numbers. The package does not renumber or alter them.

## Not included

Internal working material stays in the project repository; it is not part of this archive.

- `plan.md` — not a script, results file, or README; internal design note (DGP planning, seeds); keep it in the project repository
- `figures/` (8 files) — not part of this archive; regenerable by the scripts; pass --include-figures to publish them

## Citation

If you use this material, cite the paper and this archive ([DOI to be added]).
