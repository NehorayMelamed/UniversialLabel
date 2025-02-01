# UniversalLabeler Docker Installation Guide

---

## Introduction

Welcome to the **UniversaLabeler** Docker installation guide! Using Docker ensures a seamless setup process, isolating dependencies and providing a consistent environment across different systems.

This guide walks you through setting up **UniversaLabeler** using Docker, including GPU support, folder mapping, and troubleshooting.

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

1. **Docker**: Install Docker by following the [official guide](https://docs.docker.com/get-docker/).
2. **NVIDIA Docker Toolkit** (for GPU support): Install it using [this guide](https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab).

---

## Pulling the Pre-Built Docker Image

To get started, pull the pre-built Docker image from Docker Hub:

```bash
docker pull nehoraymelamed/universal-labeler-base:v0.1
```

This image contains all the necessary dependencies required for running UniversaLabeler.

---

## Running the Docker Container

### 1. First-Time Setup: Downloading Models and Checkpoints

If you’re running the container for the first time and need to download models, use the following command:

```bash
sudo docker run -e DOWNLOAD_MODELS=True -e MODELS_URL="https://mega.nz/file/AiFzBSDS#BqcKazpnYaS0GR4i2HqHCsenbowzr9KjeQQ9X2VPFHY" -it nehoraymelamed/universal-labeler-base:v0.1
```

This command ensures that all model weights and necessary data are downloaded inside the container.

---

### 2. Running with GPU Support

To utilize **GPU acceleration**, run the following command:

```bash
sudo docker run --gpus all -it --entrypoint bash nehoraymelamed/universal-labeler-base:v0.1
```

This grants the container access to your system’s **NVIDIA GPU**, allowing for optimal performance.

---

## Mapping Folders for Data and Output

For convenience, you can map directories from your **host machine** to the Docker container. This enables easy access to input data and storing output results.

#### Example:
- **Map a local data directory** to `/app/data` inside the container.
- **Map an output directory** to `/app/output` inside the container.

Run the following command:

```bash
sudo docker run --gpus all \
   -v /path/to/your/data:/app/data \
   -v /path/to/your/output:/app/output \
   -it nehoraymelamed/universal-labeler-base:v0.1
```

#### Explanation:
- `-v /path/to/your/data:/app/data` → Maps the **local directory** `/path/to/your/data` to `/app/data` inside the container.
- `-v /path/to/your/output:/app/output` → Maps the **local directory** `/path/to/your/output` to `/app/output` inside the container.

This setup allows the container to interact with files from the host machine, making it easier to process images and retrieve results.

---

## Dockerfile Overview

The **UniversaLabeler Docker image** is built on top of `anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04`, ensuring compatibility with deep learning frameworks and hardware acceleration.

This image includes:
- Pre-installed **PyTorch** with CUDA support.
- All necessary dependencies for **image processing, detection, and segmentation**.
- Script automation for **model downloads and inference execution**.

---

## Troubleshooting

### 1. Verify GPU Access
Run the following command inside the container to check GPU availability:

```bash
nvidia-smi
```

If the command fails or returns an error, verify that:
- **NVIDIA drivers are correctly installed**.
- **Docker is configured for GPU passthrough**.

### 2. Check Environment Variables
Ensure that environment variables are correctly passed when downloading models:

```bash
echo $MODELS_URL
```

### 3. Inspect Docker Logs
If any issue arises, check the container logs:

```bash
docker logs <container_id>
```

Replace `<container_id>` with the **actual container ID** obtained using:

```bash
docker ps -a
```

---

## Next Steps

Once your **Docker container** is set up and running, proceed to download and configure the necessary AI models.

[Go to Model Download Guide](models-download.md)

---

## Contact
For additional support, visit **Docker Hub**, or reach out to the UniversaLabeler team.

Now that installation is complete, you’re all set to start using UniversaLabeler inside a **containerized environment**! 🚀

