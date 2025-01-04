# UniversalLabeler Docker Usage Guide

---

### Prerequisites
1. **Docker Installed**:
   - Ensure Docker is installed. Refer to [Docker Installation Guide](https://docs.docker.com/get-docker/).
2. **NVIDIA Docker Toolkit Installed** (for GPU support):
   - Follow this [guide](https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab).

---

### Pulling the Pre-Built Docker Image
Pull the pre-built Docker image from Docker Hub:
```bash
docker pull nehoraymelamed/universal-labeler-base:v0.1
```

---

### Running the Docker Container

#### **1. For the First Time - Downloading Models and Checkpoints**
Run the container with the following flags to download weights:
```bash
sudo docker run -e DOWNLOAD_MODELS=True -e MODELS_URL="https://mega.nz/file/AiFzBSDS#BqcKazpnYaS0GR4i2HqHCsenbowzr9KjeQQ9X2VPFHY" -it nehoraymelamed/universal-labeler-base:v0.1
```

#### **2. Running with GPU Access**
To run the container with GPU support:
```bash
sudo docker run --gpus all -it --entrypoint bash nehoraymelamed/universal-labeler-base:v0.1
```

---

### Mapping Folders for Data and Output
To make it easier to manage data and outputs, you can map directories from the host to the container.

#### Example:
- Map a **data directory** to `/app/data` for input files.
- Map an **output directory** to `/app/output` for results.

```bash
sudo docker run --gpus all \
    -v /path/to/your/data:/app/data \
    -v /path/to/your/output:/app/output \
    -it nehoraymelamed/universal-labeler-base:v0.1
```

#### Explanation:
- `-v /path/to/your/data:/app/data`: Maps the host directory `/path/to/your/data` to `/app/data` inside the container.
- `-v /path/to/your/output:/app/output`: Maps the host directory `/path/to/your/output` to `/app/output` inside the container.

---

### Dockerfile Overview
The image is based on `anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04` and contains all required dependencies for UniversalLabeler.

---

### Entry Point Script
The container automatically runs `entrypoint.sh` to handle initialization tasks such as downloading checkpoints and performing basic tests.

```bash
#!/bin/bash

echo "### Starting Container ###"

# Check if models need to be downloaded
if [ "$DOWNLOAD_MODELS" = "True" ]; then
    echo "### Downloading models from $MODELS_URL ###"
    python setup/simply_download_pts.py "$MODELS_URL" "--mega"
    echo "### Models downloaded successfully ###"
else
    echo "### Skipping model download ###"
fi

# Run tests or start the application
if [ "$RUN_TESTS_ON_STARTUP" = "True" ]; then
    echo "### Running tests ###"
    python setup/simply_run_tests.py
else
    echo "### UniversalLabeler is ready for use ###"
fi
```

---

### Troubleshooting
1. **Verify GPU Access**:
   - Run `nvidia-smi` to confirm GPU availability.
2. **Check Variables**:
   - Ensure `MEGA_URL` is correctly set when downloading models.
3. **Use Logs**:
   - Inspect logs with `docker logs <container_id>`.

---

### Contact
For questions or issues, refer to Docker Hub or contact support.
