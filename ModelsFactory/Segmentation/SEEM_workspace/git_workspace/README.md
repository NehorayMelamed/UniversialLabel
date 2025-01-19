
# SEEM Minimal Project Documentation

## Project Overview
This project provides a minimal implementation of the **SEEM (Segment Everything Everywhere All at Once)** model, which allows for image segmentation using various types of prompts including visual and text-based prompts.

### Key Features
- **Panoptic Segmentation**
- **Semantic Segmentation**
- **Instance Segmentation**
- **Text-based Grounding**

## Usage Instructions
The project includes two primary scripts:
1. **example_usage.py** - Demonstrates how to use the SEEM model for both regular and text-based inference.
2. **seem_usage.py** - Contains core functions for loading the model, running inference, and visualizing results.

### How to Run
1. Clone this repository.
2. Ensure you have the required dependencies installed.
3. Modify the paths in the `example_usage.py` script to point to your images and model files.
4. Run the script using:
   ```bash
   python example_usage.py
   ```

### Core Functions
#### `load_model(config_path, checkpoint_path)`
Loads the SEEM model and returns the model instance, class names, and color list.

#### `infer_image(model, image)`
Runs inference on a given image for panoptic, semantic, or instance segmentation.

#### `infer_image_text_prompt(model, image, reftxt)`
Runs text-based inference using a text prompt to detect specific objects or regions.

#### `draw_regular_inference(visual, results, seg_type)`
Draws the results of regular inference for different segmentation types (panoptic, semantic, or instance).

#### `draw_text_based_results(visual, pred_masks_pos, reftxt, pred_class, colors_list)`
Draws the results of text-based inference.

## Acknowledgements
This project is based on the official SEEM repository. The full documentation and more advanced use cases can be found [here](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once).
