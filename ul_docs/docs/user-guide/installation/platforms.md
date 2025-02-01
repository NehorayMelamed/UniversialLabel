# Supported Platforms

## Introduction

UniversaLabeler supports multiple platforms, ensuring flexibility across different operating systems and hardware configurations. This section provides installation guidelines tailored to specific environments.

## System Requirements

Before proceeding, ensure your system meets the **minimum hardware and software requirements**:

- **Operating Systems:**
  - Ubuntu 20.04+ (Recommended for best performance)
  - Windows 10/11 (WSL recommended for full compatibility)
  - NVIDIA DGX (for enterprise-level GPU acceleration)

- **Hardware Requirements:**
  - NVIDIA CUDA-compatible GPU (recommended for optimal performance)
  - Minimum 16GB RAM (32GB recommended for large-scale models)
  - At least 20GB of available storage

!!! warning "Windows Users"
    For Windows users, it is **highly recommended** to use **Windows Subsystem for Linux (WSL2)** for improved compatibility. Standard Windows installation may face issues with GPU support and dependencies.

## Platform-Specific Installation Guides

### Ubuntu (Recommended)

1. **Ensure system packages are up-to-date:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
2. **Install dependencies:**
   ```bash
   sudo apt install python3 python3-venv python3-pip git -y
   ```
3. **Ensure GPU drivers and CUDA are installed:**
   ```bash
   nvidia-smi
   ```
   If CUDA is not installed, follow [NVIDIA’s installation guide](https://developer.nvidia.com/cuda-downloads).
4. **Proceed to [Environment Setup](environment-setup.md).**

### Windows (Using WSL2)

1. **Enable WSL2** (If not already enabled):
   ```powershell
   wsl --install
   ```
2. **Install Ubuntu on WSL2:**
   - Open **PowerShell** and run:
     ```powershell
     wsl --install -d Ubuntu-20.04
     ```
3. **Update system packages:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
4. **Ensure GPU support with NVIDIA CUDA for WSL2:**
   - Follow [Microsoft’s WSL2 GPU guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
5. **Proceed to [Environment Setup](environment-setup.md).**

### NVIDIA DGX Systems

1. **Ensure NVIDIA drivers are installed:**
   ```bash
   nvidia-smi
   ```

## Next Steps

- [Environment Setup](environment-setup.md) – Setting up virtual environments.
- [Package Installation](package-installation.md) – Installing required dependencies.
- [Model Download](models-download.md) – Obtaining model weights.


