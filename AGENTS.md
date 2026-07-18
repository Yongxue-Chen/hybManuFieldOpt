# Agent Instructions

## Python Environment

This project requires the conda environment **myenv**.

### Always use conda myenv when:

- Running any Python script
- Installing Python packages
- Checking package versions

### Correct usage

```bash
# Activate the environment first
conda activate myenv

# Run a script
python main_optimize.py

# Install a package
pip install <package>
```

### Never do this

```bash
python main_optimize.py       # wrong: uses system/base python if not activated
pip install <package>    # wrong: installs to wrong env if not activated
```

## Project Overview

Hybrid manufacturing continuous optimization project using PyTorch.

Main entry points:
- `main_optimize.py` — main training entry point
- `main_pre_train.py` — pre-training entry point
- `bayesian_weight_optimizer.py` — Bayesian weight optimization

## Output Language

- 输出 plan 时必须使用中文 (You must use Chinese when outputting an implementation plan).
