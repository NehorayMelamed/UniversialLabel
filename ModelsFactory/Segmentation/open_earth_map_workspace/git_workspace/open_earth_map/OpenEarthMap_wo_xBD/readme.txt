train/val/test split: train.txt, val.txt, and test.txt contain filenames for train, validation, and test sets, respectively.

Note for xBD data: the RGB images of xBD dataset are not included in the OpenEarthMap dataset.
Please download the xBD RGB images from https://xview2.org/dataset and add them to the corresponding folders.
The "xbd_files.csv" contains information about how to prepare the xBD RGB images and add them to the corresponding folders.

Structure:
- region-folder (e.g., tokyo)
    - images:
	- tokyo_1.tif
        - tokyo_2.tif
        .
	.
	.
    - labels:
	- tokyo_1.tif
        - tokyo_2.tif
        .
	.Copy-Item -Path ".\*\images\*" -Destination "path\to\destination_directory" -Recurse
	.


