import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .modeling.BaseModel import BaseModel
from .modeling import build_model
from .utils.distributed import init_distributed
from .utils.arguments import load_opt_from_config_files
from .utils.constants import COCO_PANOPTIC_CLASSES, COCO_CATEGORIES
from .modeling.language.loss import vl_similarity
import torch.nn.functional as F

# Constants
CLASSES = COCO_PANOPTIC_CLASSES

# Initialize transformation
transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC)])  # Resize image to 512 using BICUBIC interpolation

def load_model(config_path, checkpoint_path):
    """
    Load the segmentation model and prepare it for inference.

    Parameters:
    - config_path (str): Path to the model configuration file.
    - checkpoint_path (str): Path to the model checkpoint file.

    Returns:
    - model (BaseModel): Loaded segmentation model.
    - all_classes (list of str): List of all class names.
    - colors_list (list of list): List of RGB color values for each class.
    """
    # Load configuration
    opt = load_opt_from_config_files([config_path])  # Load configuration options from the file
    opt = init_distributed(opt)  # Initialize distributed processing if applicable
    opt['MODEL']['DECODER']['GROUNDING']['ENABLED'] = True  # Enable grounding task in the model

    # Build and load model
    model = BaseModel(opt, build_model(opt)).from_pretrained(checkpoint_path).eval().cuda()  # Load pre-trained model and move it to GPU

    # Prepare class information
    all_classes = [name.replace('-other','').replace('-merged','') for name in CLASSES] + ["background"]  # Clean class names
    colors_list = [color['color'] for color in COCO_CATEGORIES] + [[255, 255, 255]]
    model.model.metadata.stuff_classes = all_classes  # Assign class names to model metadata
    model.model.metadata.stuff_colors = colors_list  # Assign class names to model metadata

    # Preload text embeddings for panoptic segmentation
    with torch.no_grad():  # Disable gradient calculation
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(all_classes, is_eval=True)  # Load embeddings
        
    model.model.instance_on = True  # Enable instance segmentation

    return model, all_classes, colors_list  # Return the model, class names, and colors

def infer_image(model, image):
    """
    Run inference on a given image for panoptic, semantic, or instance segmentation.

    Parameters:
    - model (BaseModel): The loaded model for inference.
    - image (PIL.Image.Image or np.ndarray): The input image, can be either a PIL Image or a NumPy array.

    Returns:
    - results (dict): The inference results containing segmentation data.
    - image_np (np.ndarray): The original image in NumPy array format.
    """
    # Convert NumPy array to PIL image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # If input is a NumPy array, convert it to a PIL image

    # Apply transformation (resize) to the image
    image = transform(image)

    # Convert image to tensor (HxWxC -> CxHxW), and move to GPU
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().cuda()  

    # Prepare data dictionary for model input
    data = {"image": image_tensor, "height": image_tensor.shape[1], "width": image_tensor.shape[2]}  

    # Perform inference without gradients
    with torch.no_grad():
        results = model.model.evaluate([data])  # Run inference on the model

    # Convert PIL image back to NumPy array for visualization
    return results[0], np.array(image)

def draw_regular_inference(visual, results, seg_type='panoptic'):
    """
    Draw the results of regular inference for panoptic, semantic, or instance segmentation.

    Parameters:
    - visual (Visualizer): Visualizer object to handle the drawing.
    - results (dict): The inference results containing segmentation data.
    - seg_type (str): Type of segmentation ('panoptic', 'semantic', or 'instance').

    Returns:
    - result_image (PIL.Image.Image): The image with the drawn segmentation results.
    """
    if seg_type == 'panoptic':
        pano_seg, seg_info = results.get('panoptic_seg', (None, None))  # Get panoptic segmentation result
        if pano_seg is not None:
            result_image = visual.draw_panoptic_seg(pano_seg.cpu(), seg_info).get_image()  # Draw panoptic segmentation
        else:
            result_image = visual.get_image()  # Return the original image if no segmentation is found
    elif seg_type == 'semantic':
        semantic_seg = results.get('sem_seg', None)  # Get semantic segmentation result
        if semantic_seg is not None:
            semantic_seg = semantic_seg.argmax(dim=0)  # Get the most likely class for each pixel
            result_image = visual.draw_sem_seg(semantic_seg.cpu()).get_image()  # Draw semantic segmentation
        else:
            result_image = visual.get_image()
    elif seg_type == 'instance':
        instances = results.get('instances', None)  # Get instance segmentation result
        if instances is not None:
            result_image = visual.draw_instance_predictions(instances.to("cpu")).get_image()  # Draw instance segmentation
        else:
            result_image = visual.get_image()
    else:
        result_image = visual.get_image()

    return Image.fromarray(result_image)  # Return the result image as a PIL image

def infer_image_text_prompt(model, image, reftxt: str = ""):
    """
    Run text-based inference for detecting objects or regions based on a given text prompt.

    Parameters:
    - model (BaseModel): The loaded model for inference.
    - image (dict): Dictionary containing image and mask data.
    - reftxt (str): The text prompt for object grounding.

    Returns:
    - pred_masks_pos (np.ndarray): Predicted mask positions.
    - image_ori (np.ndarray): Original image in NumPy array format.
    - pred_class (torch.Tensor): Predicted class based on the text prompt.
    """
    # Apply transformation (resize) to the image
    if isinstance(image['image'], np.ndarray):
        img_data = Image.fromarray(image['image'])  # If input is a NumPy array, convert it to a PIL image
    else:
        img_data = image['image']
    image_ori = transform(img_data)
    
    # Get image width and height
    width = image_ori.size[0]
    height = image_ori.size[1]
    
    # Convert PIL image to NumPy array
    image_ori = np.asarray(image_ori)
    
    # Convert NumPy array to tensor and move to GPU
    image = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

    # Prepare data dictionary for the model
    data = {"image": image, "height": height, "width": width}
    
    # Set model tasks to use grounding only
    model.model.task_switch['spatial'] = False  
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = True
    data['text'] = [reftxt]  # Add the text prompt to the data

    batch_inputs = [data]  # Wrap data in a list for batch processing
    results, image_size, extra = model.model.evaluate_demo(batch_inputs)  # Run inference using demo evaluation mode

    pred_masks = results['pred_masks'][0]  # Get predicted masks
    v_emb = results['pred_captions'][0]  # Get visual embeddings
    t_emb = extra['grounding_class']  # Get text embeddings for grounding

    # Normalize embeddings for comparison
    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)  
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale  # Get temperature scaling for similarity
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)  # Compute similarity between visual and text embeddings
    
    matched_id = out_prob.max(0)[1]  # Get the highest matching ID
    pred_masks_pos = pred_masks[matched_id,:,:]  # Get the mask corresponding to the matched ID
    pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]  # Get the predicted class for the matched mask

    # Resize and interpolate mask back to original image size
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:height,:width] > 0.0).float().cpu().squeeze(0).numpy()

    return pred_masks_pos, image_ori, pred_class  # Return the predicted masks, original image, and predicted class

def draw_text_based_results(visual, pred_masks_pos, reftxt, pred_class, colors_list):
    """
    Draw the results of text-based inference.

    Parameters:
    - visual (Visualizer): Visualizer object to handle the drawing.
    - pred_masks_pos (np.ndarray): Predicted mask positions.
    - reftxt (str): Reference text used for grounding.
    - pred_class (torch.Tensor): Predicted class ID for the mask.
    - colors_list (list): List of RGB colors for the classes.

    Returns:
    - res (PIL.Image.Image): The image with the drawn text-based segmentation results.
    """
    # check and normalize colors_list (0-1)
    if np.max(colors_list) > 1:
        colors_list = np.array(colors_list) / 255.
    
    for mask in pred_masks_pos:
        demo = visual.draw_binary_mask(mask, color=colors_list[pred_class[0] % 133], text=reftxt)  # Draw binary mask with text label

    res = demo.get_image()  # Get final image with drawn results
    return Image.fromarray(res)  # Return result as a PIL image