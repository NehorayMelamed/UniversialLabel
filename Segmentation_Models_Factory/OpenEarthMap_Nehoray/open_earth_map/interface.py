import os
import time
import rasterio
import warnings
import numpy as np
import torch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import open_earth_map as oem

warnings.filterwarnings("ignore")

def segment_images(input_path, model_dir, model_name, output_dir, n_classes=6, device_id=3):
    # Set the CUDA device
    torch.cuda.set_device(device_id)
    
    # Prepare directories
    preds_dir = os.path.join(output_dir, "predictions_unet_test")
    overlayed_dir = os.path.join(preds_dir, 'overlayed_images')
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(overlayed_dir, exist_ok=True)
    
    # Prepare list of images to process
    if os.path.isdir(input_path):
        test_fns = [str(f) for f in Path(input_path).rglob("*.tif")]
    else:
        test_fns = [input_path]

    print("Total samples   :", len(test_fns))

    # Load dataset
    test_data = oem.dataset.OpenEarthMapDataset(
        test_fns,
        n_classes=n_classes,
        augm=None,
        testing=True,
    )

    # Load the model
    network = oem.networks.UNetFormer(in_channels=3, n_classes=n_classes)
    network = oem.utils.load_checkpoint(
        network,
        model_name=model_name,
        model_dir=model_dir,
    )
    network.eval().to(f"cuda:{device_id}")

    for idx in range(len(test_fns)):
        img, fn = test_data[idx][0], test_data[idx][2]
        print(f"Processing image {idx + 1}/{len(test_fns)}: {fn}")
        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(device_id)).squeeze(0).cpu()
        prd = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))

        fout = os.path.join(preds_dir, fn.split("/")[-1])
        with rasterio.open(fn, "r") as src:
            profile = src.profile
            prd = cv2.resize(
                prd,
                (profile["width"], profile["height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            with rasterio.open(fout, "w", **profile) as dst:
                for i in src.indexes:
                    dst.write(prd[:, :, i - 1], i)

        pred = plt.imread(fout)
        original_image = plt.imread(fn)
        added_image = cv2.addWeighted(original_image, 0.6, pred, 0.4, 0)

        plt.imsave(os.path.join(overlayed_dir, f"{fn.split('/')[-1].split('.')[0]}_overlayed.png"), added_image)

    print(f"Processing completed. Results saved in {preds_dir} and {overlayed_dir}.")

if __name__ == "__main__":
    start_time = time.time()

    input_path = "/raid/open_earth_map/xbd/images/aachen_20.tif"  # Or a directory path
    model_dir = "/raid/open_earth_map/outputs_300_epochs"
    model_name = "u_former_best.pth"
    output_dir = "interface_output"

    segment_images(input_path, model_dir, model_name, output_dir)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
