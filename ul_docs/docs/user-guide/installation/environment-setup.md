# Environment Setup

## Introduction

Setting up a dedicated environment ensures **UniversaLabeler** runs smoothly without conflicting with other dependencies. We recommend using a virtual environment, such as **venv** or **Conda**, to isolate the project dependencies.

## Virtual Environment Options

UniversaLabeler supports two primary environment configurations:

1. **Python Virtual Environment (venv) – Recommended for most users.**
2. **Conda Environment – Preferred for advanced users and package compatibility management.**

---

## Setting Up a Python Virtual Environment (venv)

1. **Ensure Python 3.9+ is installed:**
   ```bash
   python3 --version
   ```
   If not installed, follow the [Python installation guide](https://www.python.org/downloads/).

2. **Create a virtual environment:**
   ```bash
   python3 -m venv ul_env
   ```

3. **Activate the virtual environment:**
   - **Linux/macOS:**
     ```bash
     source ul_env/bin/activate
     ```
   - **Windows:**
     ```powershell
     ul_env\Scripts\activate
     ```

4. **Verify activation:**
   Running the following command should show `(ul_env)` at the start of the command line:
   ```bash
   which python
   ```

5. **Proceed to [Package Installation](package-installation.md).**

---

## Setting Up a Conda Environment

1. **Ensure Conda is installed:**
   - If not installed, download it from [Anaconda](https://www.anaconda.com/products/distribution) or install Miniconda:
     ```bash
     curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     bash miniconda.sh
     ```

2. **Create a new Conda environment:**
   ```bash
   conda create --name ul_env python=3.9 -y
   ```

3. **Activate the Conda environment:**
   ```bash
   conda activate ul_env
   ```

4. **Verify activation:**
   ```bash
   which python
   ```

5. **Proceed to [Package Installation](package-installation.md).**

---

## Troubleshooting

!!! warning "Common Issues"
    - **Python not found:** Ensure Python is installed and accessible in the system path.
    - **Virtual environment activation fails:** Use the absolute path when running activation commands.
    - **Package conflicts in Conda:** Try running `conda update --all` before installing additional dependencies.

## Next Steps

- [Package Installation](package-installation.md) – Install required dependencies.
- [Model Download](models-download.md) – Obtain pre-trained models for inference.

