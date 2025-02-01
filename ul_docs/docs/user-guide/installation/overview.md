# Installation Overview

## Introduction

This section provides an overview of the installation process for **UniversaLabeler**. Multiple installation methods are available depending on your use case, including standard installation, Docker-based setup, and advanced configurations.

## Installation Methods

### 1. Standard Installation
The standard installation method is recommended for most users. It involves setting up a virtual environment and installing dependencies using `pip`.

- Suitable for local development and deployment.
- Requires Python 3.9 or 3.10.
- Uses `pip` for package management.

[Go to Standard Installation Guide](environment-setup.md)

### 2. Docker Installation
For users who prefer **containerized environments**, a pre-configured Docker image is available.

- Ensures compatibility with all dependencies.
- Recommended for cloud-based applications and isolated environments.

[Go to Docker Installation Guide](docker-installation.md)

### 3. Advanced Installation Options
Additional methods such as **Google Colab, REST API, and Conda environments** are under development.

- Suitable for specialized use cases.
- Contact support for early access.

## Prerequisites
Before proceeding with the installation, ensure that your system meets the following requirements:

- **Operating System**: Ubuntu 20.04+, Windows 10/11 (WSL recommended), NVIDIA DGX.
- **GPU Support**: NVIDIA CUDA-compatible GPU for optimal performance.
- **Python Version**: 3.9 or 3.10.
- **Git Installed**: Required for cloning the repository.

!!! warning "Important Warning"
    Running without a GPU may result in significantly slower performance.

## Next Steps

- [Platform-Specific Installation](platforms.md) – Configure your OS for installation.
- [Environment Setup](environment-setup.md) – Creating a virtual environment.
- [Package Installation](package-installation.md) – Installing dependencies.
- [Downloading Models](models-download.md) – Obtaining required AI model weights.
