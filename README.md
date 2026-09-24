# Constrained martingale transport: numerical examples

Python code accompanying **A one-sided constrained martingale transport between two uniform laws**, by **Erhan Bayraktar and Xin Zhang**.

The paper studies martingale couplings of `X ~ Uniform[-1, 1]` and `Y ~ Uniform[-2, 2]` subject to `Y >= X - k`. The scripts evaluate the coupling formulas, generate numerical examples for the cubic cost `E[(Y-X)^3]`, and compare them with independently optimized finite-dimensional linear programs. The LP uses only the marginal, martingale, and support constraints; the analytic formulas are used afterward for comparison.

## Requirements and quick start

Use **Python 3.10 or later**. The formula, table, explanatory-data, and optional rational-verification scripts use only the Python standard library. Only the LP script requires NumPy and SciPy (which includes the HiGHS solver).

Run these commands from the repository directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python numerics/generate_data.py
python numerics/generate_explanatory_figures.py
python numerics/generate_comparison_table.py
python numerics/discrete_lp.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. If you do not need the LP comparison, omit the package installation and the last command. The scripts locate their output directories relative to their own files and may also be invoked from another working directory. Rerunning them replaces their generated outputs.

## Scripts and outputs

| Script in `numerics/` | Purpose | Generated output |
| --- | --- | --- |
| `generate_data.py` | Evaluate the lower and upper transport maps, target-marginal checks, cubic values, and transition parameters | `figures/support_k15.dat`, `support_k25.dat`, `marginal_k15.dat`, `marginal_k25.dat`, `density_decomposition_k25.dat`, `cubic_value.dat`; `numerics/generated_values.tex` |
| `generate_explanatory_figures.py` | Evaluate curves used to illustrate the construction | `figures/predecessor_support.dat`, `middle_displacements.dat`; `numerics/generated_explanatory_values.tex` |
| `generate_comparison_table.py` | Compute cubic costs and mass on the binding lower branch for eight values of `k` | `numerics/comparison_table_rows.tex` |
| `discrete_lp.py` | Solve the independent LP for `k = 1.5, 2.5` at source-grid sizes `N = 20, 40, 80, 120` | `figures/lp_convergence_k15.dat`, `lp_convergence_k25.dat`, `lp_support_k15.dat`, `lp_support_k25.dat`; `numerics/generated_lp_values.tex` |
| `verify_optimality_bound.py` | Optional historical scalar-inequality verification using exact rational intervals | Console report only |

Paths in each table cell are relative to the repository root; filenames following a directory prefix remain in that directory. Generated `.dat` files are whitespace-delimited with a header row. They are **plot data, not rendered figures**. The `.tex` files provide numerical macros and table rows for the manuscript. This repository contains the numerical code, not the manuscript or its LaTeX plotting templates. Generated files are omitted from version control and are recreated by the commands above.

## Scope and checks

`generate_data.py` implements `SmallRegime` for `1 < k < 2` and `LargeRegime` for `2 <= k <= 3`. The binding branch is evaluated with the principal real Lambert W function, implemented locally for the range needed here; other scalar inversions use bisection. At `k = 1`, the coupling is the equal-weight pair `Y = X - 1` and `Y = X + 1`; it is not evaluated by `SmallRegime`. For `k >= 3`, the constraint is inactive and the `k = 3` maps apply.

The scripts check martingale and support residuals, the target second moment and CDF approximation, four-decimal stability of table values under quadrature refinement, and LP equality residuals. The LP support files use `N = 80`; the convergence files contain all four grids. Different SciPy/HiGHS versions may select different optimizers when a discrete problem has more than one solution.

These numerical checks illustrate and test the constructions; they do not replace the paper's analytic optimality proof. In particular, the optional command

```sh
python numerics/verify_optimality_bound.py
```

checks a scalar inequality from an earlier proof approach, using fractions and interval bounds rather than floating-point arithmetic. **The current analytic proof does not depend on this supplemental calculation.**

## Reproducibility reference

All five scripts were run successfully with **Python 3.14.7, NumPy 2.5.3, and SciPy 1.18.1**. Representative results are:

| `k` | Cubic cost from the formulas | LP cubic cost, `N = 120` |
| --- | ---: | ---: |
| 1.5 | -0.673258643140 | -0.673153744278 |
| 2.5 | -1.446789916088 | -1.446621929652 |

The target second moment was `1.3333333333` in both cases (the exact value is `4/3`). Every built-in check passed, including LP equality residuals below `1e-10` and agreement of the comparison-table values to four decimals under quadrature refinement. The optional rational check accepted 3,237 intervals, with maximum subdivision depth 14 and exact total interval length `4/5`.
