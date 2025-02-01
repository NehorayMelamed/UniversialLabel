import os
import sys
import zipfile
import requests
from tqdm import tqdm
from mega import Mega

# sys.path.append("/home/nehoray/PycharmProjects/UniversaLabeler")
# Constants for default paths
from common.general_parameters import WEIGHTS_PATH
DOWNLOAD_PTS_FILE_PATH = "downloaded_file.zip"


def download_file(url, output_path):
    """
    Download a file from the given URL and save it to the specified output path with a progress bar.
    """
    print(f"Downloading file from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(output_path, 'wb') as file, tqdm(
        desc="Progress",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))

    print(f"File downloaded successfully to {output_path}")


def download_file_from_mega(url, output_path):
    """
    Download a file from a Mega URL and save it to the specified output path.
    """
    print(f"Downloading file from Mega URL: {url}")
    mega = Mega()
    m = mega.login()  # Login as anonymous user
    file = m.download_url(url, dest_filename=output_path)
    print(f"File downloaded successfully to {output_path}")


def extract_zip_file(zip_file_path, extract_to):
    """
    Extract a ZIP file to the specified directory.
    """
    print(f"Extracting {zip_file_path} to {extract_to}")
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to {extract_to}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and extract model weights.")
    parser.add_argument("url", type=str, help="The URL to download the file from")
    parser.add_argument(
        "--mega", action="store_true", help="Use this flag if the URL is a Mega link"
    )
    parser.add_argument(
        "--output", type=str, default=DOWNLOAD_PTS_FILE_PATH, help="Path to save the downloaded file"
    )
    parser.add_argument(
        "--extract_to", type=str, default=WEIGHTS_PATH, help="Directory to extract the contents of the ZIP file"
    )

    args = parser.parse_args()

    try:
        # Download the file
        if args.mega:
            download_file_from_mega(args.url, args.output)
        else:
            download_file(args.url, args.output)

        # Extract the ZIP file
        extract_zip_file(args.output, args.extract_to)

    except Exception as e:
        print(f"Failed to download or extract files: {str(e)}")
