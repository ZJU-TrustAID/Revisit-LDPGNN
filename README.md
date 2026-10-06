## Revisiting Locally Differentially Private Graph Neural Networks

> **Accepted at the IEEE Symposium on Security and Privacy (IEEE S&P 2027).**

**Authors:** Zhewen Hu (Zhejiang University), Zhikun Zhang* (Zhejiang University), Bo Sun (Zhejiang University), Quan Yuan (Zhejiang University), and Yunjun Gao (Zhejiang University).

\* Corresponding author.

## Quick start

Create the isolated Conda environment and install all dependencies:

```bash
bash scripts/setup_env.sh
conda activate revisit-ldpgnn
```

The setup script creates the environment with Python 3.10 and installs the PyTorch, PyG, notebook, and experiment dependencies. Then open the notebooks:

```bash
python -m jupyter notebook notebooks/
```

## Which notebook to run

### Notebook 1: Direct results

Notebook 1 is the fastest entry point. It uses the experiment result data that we have already produced and directly generates the same figures and the corresponding tables shown in the paper. It does not run training or hyperparameter search.

Open `notebooks/notebook_1_direct.ipynb` from the Jupyter interface.

### Notebook 2: Fixed hyperparameters

Notebook 2 reruns all training-based figure points and baselines using the supplied fixed hyperparameters.

Edit the configuration cell to set `AEC_EXECUTE = True` when running the experiments. With `False`, the notebook only prints the execution plan. GPU selection and per-GPU concurrency are configured in the same cell.

In the configuration cell of Notebooks 2 and 3, both `AEC_GPU_IDS` and `AEC_MAX_PARALLEL_PER_GPU` default to `'default'`. This selects all visible GPUs and sets the maximum simultaneous training tasks on each GPU to its total VRAM divided by 6 GiB, rounded down.

To choose GPUs and concurrency manually, edit those variables in the same cell:

```python
AEC_GPU_IDS = [0, 2]
AEC_MAX_PARALLEL_PER_GPU = [3, 5]
```

This example selects GPU indices 0 and 2 (the first and third GPUs visible to the notebook), allowing up to 3 training tasks on GPU 0 and 5 on GPU 2 at the same time. Setting `AEC_MAX_PARALLEL_PER_GPU = 3` applies a limit of 3 tasks to every selected GPU.

Open `notebooks/notebook_2_fixed_hparams.ipynb` from the Jupyter interface and run the configuration cell first.

### Notebook 3: Hyperparameter search

Notebook 3 runs our hyperparameter search scripts and has two modes:

- **claim-coverage scaled mode** includes the core results: Figure 1 and Table 4 display Cora and Facebook, and Figure 6 displays GraphSAGE.
- **full mode** runs the complete experiment configuration and is the complete reproduction path.

Both modes run the same YAML-driven hyperparameter search pipeline used by our experiments. They produce the hyperparameters needed for the data points rerun by Notebook 2, and then generate the corresponding figures and tables.

The default mode is `claim-coverage scaled`. Set `AEC_MODE = "full"` in the configuration cell when manually running the complete search.

Each notebook displays the generated PNG figures and Table 4/Table 6 directly in the notebook.

PNG and PDF files are saved under `outputs/`. Notebook 3 ends with a check of figure/table data completeness and statistics. You can also run it from the repository root:

```bash
python -m scripts.aec.validate_outputs --mode scaled
```

Use `--mode full` for a full search.

Temporary GPU memory exhaustion, other transient runtime errors, or an interrupted run may leave some training tasks unfinished. Rerun the same code cell with the same experiment settings to continue. Completed training results are reused automatically, and failed or unfinished tasks are rerun before the figure or table is generated.

The data completeness check before rendering also reports missing experiment results, helping you quickly identify incomplete runs. If it reports missing data points, rerun the same code cell to retry failed or unfinished tasks and complete the missing experiments.

## Two ways to run the notebooks

### Run manually in Jupyter

Open the notebook in Jupyter and run its configuration cell first, followed by the setup cell and the figure/table cells one by one. This is useful when you want to inspect each generated figure or change the GPU settings between runs.

```bash
python -m jupyter notebook notebooks/
```

### Run automatically from the command line

Use `nbconvert` to execute a notebook from top to bottom without opening the notebook UI. The executed notebook is written back to the same path.

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_1_direct.ipynb
```

Before running Notebook 2, set `AEC_EXECUTE = True` in its configuration cell if you want to start the fixed-hyperparameter reruns. Then execute the notebook:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_2_fixed_hparams.ipynb
```

For Notebook 3, set `AEC_EXECUTE = True` and choose `AEC_MODE = "scaled"` or `AEC_MODE = "full"` in its configuration cell, then execute:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/notebook_3_full_search.ipynb
```
