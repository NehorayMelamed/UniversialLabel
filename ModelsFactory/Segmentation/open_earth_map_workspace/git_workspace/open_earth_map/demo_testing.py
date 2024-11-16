import os
import sys
sys.path.append(os.path.dirname(__file__))
import time
import rasterio
import warnings
import numpy as np
import torch
import cv2
import open_earth_map as oem
from pathlib import Path
import matplotlib.pyplot as plt


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.set_device(3)


warnings.filterwarnings("ignore")
if __name__ == "__main__":
    start = time.time()

    OEM_DATA_DIR = "/raid/open_earth_map/OpenEarthMap_wo_xBD"
    TEST_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    N_CLASSES = 6
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PREDS_DIR = "predictions_unet_test"
    OVERLAYED_DIR = os.path.join(PREDS_DIR, 'overlayed_images')
    os.makedirs(PREDS_DIR, exist_ok=True)
    os.makedirs(OVERLAYED_DIR, exist_ok=True)

    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)][:2]
    test_fns = [str(f) for f in fns if f.name in np.loadtxt(TEST_LIST, dtype=str)]

    print("Total samples   :", len(fns))
    print("Testing samples :", len(test_fns))

    test_data = oem.dataset.OpenEarthMapDataset(
        fns,
        n_classes=N_CLASSES,
        augm=None,
        testing=True,
    )

    # network = oem.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    network = oem.networks.UNetFormer(in_channels=3, n_classes=N_CLASSES)

    network = oem.utils.load_checkpoint(
        network,
        model_name="u_former_best.pth",
        model_dir="/raid/open_earth_map/outputs_300_epochs",
        # model_name="u-efficientnet-b4_s0_CELoss_pretrained.pth",
        # model_dir="/raid/open_earth_map/outputs",
    )

    network.eval().to("cuda:3")
    for idx in range(len(fns)):
        img, fn = test_data[idx][0], test_data[idx][2]
        print("current_img: " + str(idx))
        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        prd = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))

        fout = os.path.join(PREDS_DIR, fn.split("/")[-1])
        with rasterio.open(fn, "r") as src:
            profile = src.profile
            prd = cv2.resize(
                prd,
                (profile["width"], profile["height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            with rasterio.open(fout, "w", **profile) as dst:
                for idx in src.indexes:
                    dst.write(prd[:, :, idx - 1], idx)

        pred = plt.imread(fout)
        original_image = plt.imread(fn)
        added_image = cv2.addWeighted(original_image, 0.6, pred, 0.4, 0)

        plt.imsave(os.path.join(OVERLAYED_DIR, f"{fn.split('/')[-1].split('.')[0]}_overlayed.png"), added_image)