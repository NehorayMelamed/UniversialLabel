# Use the prebuilt anibali/pytorch image with CUDA 11.8
FROM anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app/UniversaLabeler"


# Switch to root user for installing system dependencies
USER root

# Install additional system dependencies
RUN echo "### Installing additional system dependencies ###" && \
    apt-get update && \
    apt-get install -y tzdata wget curl git libopencv-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/UniversaLabeler

# Copy the project files into /app/UniversaLabeler
COPY . /app/UniversaLabeler

# Switch back to non-root user

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Entry point script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


# Default command
ENTRYPOINT ["/entrypoint.sh"]
