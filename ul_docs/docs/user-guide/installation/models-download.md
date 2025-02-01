# Model Download

## Introduction

To utilize **UniversaLabeler**, pre-trained model checkpoints must be downloaded and placed in the appropriate directory. This section guides you through the steps for retrieving and setting up these files.

---

## Step 1: Download Model Checkpoints

### **Option 1: Automatic Download (Recommended)**

Run the provided script to automatically fetch the required models:

```bash
python setup/simply_download_pts.py mega.nz/file/AiFzBSDS#BqcKazpnYaS0GR4i2HqHCsenbowzr9KjeQQ9X2VPFHY --mega
```

This script will:
- Download the necessary model weights.
- Verify the integrity of each file.
- Place them in the correct directory.

### **Option 2: Manual Download**

If you prefer to download models manually, use the provided links:

- [MEGA Repository](https://mega.nz/file/AiFzBSDS#BqcKazpnYaS0GR4i2HqHCsenbowzr9KjeQQ9X2VPFHY)

After downloading, extract the files into the following directory:

`UniversaLabeler/common/weights`


!!! warning "Ensure Correct Placement"
    Model files must be located in **UniversaLabeler/common/weights/** to be detected properly.

---

## Step 2: Verify Installation

To check that the models were installed correctly, please visit the UniversaLabeler/common/weights:


---

## Next Steps

- [Basic Usage](../usage/basic-usage.md) – Learn how to run an initial inference.
- [Advanced Configuration](../usage/get-started-usage.md) – Customize model settings and priorities.


