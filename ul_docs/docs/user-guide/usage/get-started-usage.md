# Getting Started with UniversaLabeler

## Project Scope

After successfully installing the package, downloading all required models, and passing the startup tests, we can now proceed with using **UniversaLabeler** and its components.

!!! warning "Powerful tool"
    You are now using a **powerful and complex tool** as part of a large-scale project. Please be mindful and responsible while using it.

### Understanding the System

To effectively utilize the project, it is important to grasp its **core structure**. You can refer to the **development section** for deeper insights into its architecture and design.

### The UL Class

At the heart of **UniversaLabeler** lies the `UL` class, which serves as the central interface for executing various labeling tasks. It provides a structured approach to handling **object detection**, **segmentation**, and **captioning** workflows.

Within this core class, we have the following specialized implementations:

- `ULDetection`: Handles **bounding box-based** object detection.
- `ULSegmentation`: Manages segmentation tasks with **semantic, instance, and panoptic** segmentation support.
- `ULCaptioning`: Supports **image description, classification, and object labeling** tasks.

Each of these classes provides a seamless interface for interacting with different AI models while incorporating **pre-processing** and **post-processing** pipelines.

### How It Works

Each `UL` implementation uses a **Factory Interface** that dynamically initializes models based on user preferences. Additionally, it integrates with **algorithmic enhancement layers**, such as:

- **Pre-Processing Mechanisms:** Optimizing input images before inference.
- **Post-Processing Enhancements:** Refining results using techniques like **Non-Maximum Suppression (NMS)**.
- **Captioning Pipelines:** Utilizing **LLM-based reasoning** for contextual image interpretation.

This modular approach ensures flexibility and configurability while maintaining a user-friendly interface.

---

## Next Steps

- [**Basic Usage**](basic-usage.md) - Learn how to use UniversaLabeler with practical examples.
- [**UL Detection**](../../development/architecture.md)(ul-detection.md) - A deep dive into detection workflows.
- [**UL Segmentation**](ul-segmentation.md) - Explore segmentation capabilities.
- [**UL Captioning**](ul-captioning.md) - Learn about AI-powered image captioning.


