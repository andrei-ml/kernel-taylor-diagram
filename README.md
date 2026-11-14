# Nonlinear kernel-based extensions of the Taylor diagram

This repository provides a set of inner products for plotting Taylor diagrams.
The classic Taylor diagram shows in one plot the centered RMSE, the correlation
and the standard deviations of two datasets, X (reference) and Y (tested). All these quantities
are derived from one inner product between X and Y. Here we generalize that inner product with
kernels, so that the diagram can also capture nonlinear, structural and distributional
similarity.

---

## Types of inner product implemented

| Type             | Compares                               | Insensitive to                     |
|------------------|----------------------------------------|------------------------------------|
| `classic`        | the samples pair by pair, linearly     | how the error spreads over samples |
| `kernel_classic` | paired samples, with a tolerance σ     | the size of errors beyond σ        |
| `kernel_hsic`    | the internal structure of each dataset | rotations, reflections and shifts  |
| `kernel_mmd`     | the distributions of the points        | the order of the samples           |

- *Classic*: the covariance between the datasets.
- *Kernel classic*: the **nonlinear** covariance between the paired samples mapped to a
  reproducing kernel Hilbert space (RKHS).
- *Kernel HSIC*: the similarity between the centered kernel matrices of X and Y, based on the
  Hilbert-Schmidt Independence Criterion. The correlation is the centered kernel alignment.
- *Kernel MMD*: a **probabilistic** similarity based on the maximum mean discrepancy, comparing
  the distributions of X and Y regardless of sample order ([Wickstrøm et al., 2022](https://doi.org/10.1007/978-3-031-17030-0_10)).


---

## Installation

**With module installation.** Install the modules `kernel_taylor_diagram` and `taylor_diagram_plot` with
their dependencies, to import them in your own scripts:

```bash
pip install git+https://github.com/andrei-ml/kernel-taylor-diagram.git
```

**Without module installation.** Clone the repository and install only the dependencies:

```bash
git clone https://github.com/andrei-ml/kernel-taylor-diagram.git
cd kernel-taylor-diagram
pip install -r requirements.txt
```

Then see the import example in the notebook [demo1_synthetic_example.ipynb](demo1_synthetic_example.ipynb).

**To run the notebook**, additionally install Jupyter:  
```bash
pip install jupyter
```

---

## Quickstart

```python
import kernel_taylor_diagram as ktd

# X: reference, Y: tested, both numpy arrays of shape [n_samples, n_features]
n, d = X.shape
sigma = X.std(axis=0).mean() * n**(-1. / (d + 4.))       # Scott's rule for the kernel width

estimator = ktd.TaylorEstimator('kernel_mmd', normalized=True, gamma=1. / (2. * sigma**2))
res = estimator.compute(X, Y)
# res is a dict with keys 'std_X', 'std_Y', 'corr', 'rmse'
```

The notebook [demo1_synthetic_example.ipynb](demo1_synthetic_example.ipynb) introduces the
Taylor diagram, kernels and centered kernel matrices, and compares the four inner products on
synthetic 2D data. Run it from the repository root.

---

## Citation and license

If you use this code, please cite:

Gavrilov, A., Mankovich, N., Link, M., Huang, F., and Camps-Valls, G.: Kernel Taylor Diagram for
Earth System Model Evaluation, EGU General Assembly 2026, EGU26-7678,
https://doi.org/10.5194/egusphere-egu26-7678, 2026.

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.

Developed within AI4PEX, funded by the European Union's Horizon Europe programme under grant
agreement no. 101137682.