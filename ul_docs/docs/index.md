# UniversaLabeler

Welcome to the **UniversaLabeler Documentation**.

UniversaLabeler offers an innovative approach to enriching information and labeling images. This is achieved by refining the image processing pipeline and utilizing advanced AI models. The project also enables the creation of new and adaptable configurations through a modular and user-friendly development framework.

## Quick Overview

UniversaLabeler processes an image through **three main layers**, each consisting of multiple sub-layers of data refinement until the final output is reached:

![Processing Flow](assets/universal_labeler_flow.png)

## Pre-Processing Image Layer

This layer applies both **classic algorithms and deep learning techniques** to refine the image before model execution. The goal is to prepare the image optimally by selecting suitable models, parameters, and enhancement techniques. These steps occur automatically or based on user preferences to ensure that each model receives a properly adjusted image. This leads to more accurate and informative final results.

## Deep Learning Models Prediction Layer

The three primary AI-driven image processing tasks handled in this layer are:

- **Detection**  
- **Segmentation**  
- **Captioning**  

Each of these tasks contains different subtasks and methodologies. Users can explore them in detail:

- **[Detection](deep-learning/detection.md)** – Object recognition with multiple input modes (free prompt, zero-shot, predefined classes).
- **[Segmentation](deep-learning/segmentation.md)** – Covers **semantic, instance, and panoptic** segmentation.
- **[Captioning](deep-learning/captioning.md)** – Supports **image description, classification, and object tagging**.

## Supported Models

Each model below serves a specific task in the pipeline. Clicking on a model name will redirect you to its official repository.

| Model Name | Input Type | Model Type | Repository |
|------------|------------|-------------|-------------|
| [GrouDino](https://github.com/IDEA-Research/GroundingDINO) | Prompt | Detection | [Repository](https://github.com/IDEA-Research/GroundingDINO) |
| [Yolo World](https://github.com/AILab-CVC/YOLO-World) | Prompt | Detection | [Repository](https://github.com/AILab-CVC/YOLO-World) |
| Yolo Alfred | Classes | Detection | *Internal Model* |
| [Waldo](https://github.com/stephansturges/WALDO) | Prompt | Detection | [Repository](https://github.com/stephansturges/WALDO) |
| [OpenEarthMap](https://github.com/bao18/open_earth_map) | Classes | Segmentation | [Repository](https://github.com/bao18/open_earth_map) |
| [SAM2](https://github.com/facebookresearch/sam2) | Bbox/None | Segmentation | [Repository](https://github.com/facebookresearch/sam2) |
| [Google Vision](https://cloud.google.com/vision) | Prompt | Detection | [Repository](https://cloud.google.com/vision) |
| [Dino-X](https://github.com/IDEA-Research/DINO-X-API) | Prompt | Detection & Segmentation | [Repository](https://github.com/IDEA-Research/DINO-X-API) |
| [OpenGeos](https://github.com/opengeos/segment-geospatial) | Prompt | Detection | [Repository](https://github.com/opengeos/segment-geospatial) |
| [TREX](https://github.com/cisco-system-traffic-generator/trex-core) | Reference | Detection | [Repository](https://github.com/cisco-system-traffic-generator/trex-core) |

## Post-Processing Image Layer


This layer integrates all obtained results and refines the conclusions. One of the key mechanisms in this stage is **NMS (Non-Maximum Suppression)**, which merges overlapping detection results and enhances consistency. Additionally, this layer ensures that final outputs are exported in user-defined formats for visualization or further processing.

## Summary

UniversaLabeler provides a **framework** that simplifies the process of passing images through **a series of enhancement, analysis, and post-processing steps**, all configurable based on user and system-defined logic.

- [Learn more about system architecture](development/architecture.md)
- [How to integrate your own models](development/model-integration.md)

> **"Every SOTA model excels at its dedicated task. Our strength lies in combining multiple SOTA models from different domains to tackle more complex challenges."**

This project also includes an external UI for better accessibility. See the [UI Usage Guide](user-guide/usage/basic-usage.md) for more details.

---

## Next Steps

- [Installation Guide](user-guide/installation/overview.md) – Setting up UniversaLabeler.
- [Basic Usage](user-guide/usage/basic-usage.md) – Running your first image processing pipeline.
- [Developer Guide](development/architecture.md) – Understanding the system architecture and extending functionalities.
- [API Reference](api/api-reference.md) – Learn about the available API endpoints.

---

~=
  <a href="contact.md" style="color: lightblue;">⬅ Previous: Contact</a>
  <a href="user-guide/installation/overview.md" style="color: lightblue;">Next: Installation Overview ➡</a>
</div>
