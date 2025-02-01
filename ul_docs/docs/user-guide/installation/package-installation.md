# Package Installation

## Introduction

Once the virtual environment is set up, the next step is to install the necessary dependencies for **UniversaLabeler**. This ensures all required libraries are correctly configured for smooth operation.

---

## Installing Required Packages

### Step 1: Ensure the Virtual Environment is Activated
Before installing packages, ensure that your virtual environment is **activated**:

- **For venv (Linux/macOS):**
  ```bash
  source ul_env/bin/activate
  ```
- **For venv (Windows):**
  ```powershell
  ul_env\Scripts\activate
  ```
- **For Conda:**
  ```bash
  conda activate ul_env
  ```

### Step 2: Install PyTorch Manually

Since PyTorch is not included in `requirements.txt`, install it manually based on your system configuration:

1. Visit the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).
2. Choose the appropriate configuration based on your system (CUDA, CPU, etc.).
3. Install using the provided command, for example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
## Verifying the pytorch Installation

Run the following command to check if everything is installed correctly:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Expected output:
- **`True`** → CUDA (GPU) is available.
- **`False`** → Troubleshoot it.
!!! warning "Running on CPU - not recommended but possible in some cases""




!!! note "TODO"
    In the next release, PyTorch will be installed automatically as part of the `requirements.txt` dependencies.

---

### Step 3: Install Other Required Python Packages

Run the following command to install all additional dependencies:

```bash
pip install -r requirements.txt
```

!!! warning "Important Notice"
    Ensure that your **pip** is up-to-date before installing dependencies to prevent compatibility issues:
    ```bash
    pip install --upgrade pip
    ```

---


## Troubleshooting

!!! warning "Common Issues"
    - **Package installation failure:** Ensure Python and pip are correctly installed.
    - **Dependency conflicts:** Run `pip check` to identify issues.
    - **PyTorch GPU not detected:** Ensure CUDA is installed and drivers are up to date.

---

## Next Steps

- [Model Download](models-download.md) – Download and configure model checkpoints.
- [Basic Usage](../usage/basic-usage.md) – Running your first image processing pipeline.

---

