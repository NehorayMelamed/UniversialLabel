import os
import time
import warnings
import numpy as np
import torch
import sys
# sys.path.append("/raid/open_earth_map/open_earth_map")
import open_earth_map
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
warnings.filterwarnings("ignore")
y_score = {} # loss history
y_score['train'] = []
y_score['val'] = []
y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="score")
ax1 = fig.add_subplot(122, title="loss")

def save_training_results(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_score['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_score['val'], 'ro-', label='val')

    ax1.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()

    fig.savefig( os.path.join('outputs/train.jpg'))

if __name__ == "__main__":
    start = time.time()

    OEM_DATA_DIR = "/raid/open_earth_map/xbd"
    # TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
    TRAIN_LIST = "/raid/open_earth_map/OpenEarthMap_wo_xBD/train.txt"
    VAL_LIST = "/raid/open_earth_map/OpenEarthMap_wo_xBD/test.txt"
    # VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    IMG_SIZE = 512
    N_CLASSES = 6
    LR = 0.001
    BATCH_SIZE = 16
    NUM_EPOCHS = 450
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "outputs_300_epochs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    train_fns = [str(f) for f in fns if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    val_fns = [str(f) for f in fns if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    print("Total samples      :", len(fns))
    print("Training samples   :", len(train_fns))
    print("Validation samples :", len(val_fns))

    train_augm = torchvision.transforms.Compose(
        [
            open_earth_map.transforms.Rotate(),
            open_earth_map.transforms.Crop(IMG_SIZE),
        ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            open_earth_map.transforms.Resize(IMG_SIZE),
        ],
    )

    train_data = open_earth_map.dataset.OpenEarthMapDataset(
        train_fns,
        n_classes=N_CLASSES,
        augm=train_augm,
    )

    val_data = open_earth_map.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=val_augm,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
    )

    network = open_earth_map.networks.UNetFormer(in_channels=3, n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = open_earth_map.losses.JaccardLoss()

    max_score = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch + 1}")

        train_logs = open_earth_map.runners.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=DEVICE,
        )

        valid_logs = open_earth_map.runners.valid_epoch(
            model=network,
            criterion=criterion,
            dataloader=val_data_loader,
            device=DEVICE,
        )

        epoch_score = valid_logs["Score"]
        if epoch % 20 == 0:
            max_score = epoch_score
            open_earth_map.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name=f"u_former_{epoch}.pth",
                output_dir=OUTPUT_DIR,
            )
        if max_score < epoch_score:
            max_score = epoch_score
            open_earth_map.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name="u_former_best.pth",
                output_dir=OUTPUT_DIR,
            )
        y_score['train'].append(train_logs["Score"])
        y_score['val'].append(valid_logs["Score"]) 
        y_loss['train'].append(train_logs['Loss'])
        y_loss['val'].append(valid_logs['Loss']) 
        
        save_training_results(epoch)
        torch.cuda.empty_cache()

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
