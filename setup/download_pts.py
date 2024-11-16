import os
import zipfile
import base64
import hashlib
from mega import Mega
from cryptography.fernet import Fernet

from common.general_parameters import ENCRYPTED_UTL_PATH, SECRET_KEY_PATH, WEIGHTS_PATH, DOWNLOAD_PTS_FILE_PATH

def load_secret_key_from_file(file_path=SECRET_KEY_PATH):
    with open(file_path, 'r') as key_file:
        return key_file.read().strip()


def load_encrypted_url_from_file(file_path=ENCRYPTED_UTL_PATH):
    with open(file_path, 'r') as url_file:
        return url_file.read().strip()

def derive_encryption_key(secret_key):
    return base64.urlsafe_b64encode(hashlib.sha256(secret_key.encode()).digest())

def decrypt_url(secret_key, encrypted_url):
    encryption_key = derive_encryption_key(secret_key)
    fernet = Fernet(encryption_key)
    decrypted_url = fernet.decrypt(encrypted_url.encode())
    return decrypted_url.decode()

def download_file_from_mega(url, output_path=DOWNLOAD_PTS_FILE_PATH):
    mega = Mega()
    m = mega.login()  # Login as anonymous user
    file = m.download_url(url, dest_filename=output_path)
    print(f"File downloaded successfully to {output_path}")

def extract_zip_file(zip_file_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files extracted to {extract_to}")

if __name__ == "__main__":
    # Load the secret key from file
    secret_key = load_secret_key_from_file()

    # Load the encrypted URL from file
    encrypted_url = load_encrypted_url_from_file()

    # Decrypt the URL
    try:
        decrypted_url = decrypt_url(secret_key, encrypted_url)
        print(f"Decrypted URL: {decrypted_url}")

        # Download the file
        download_file_from_mega(decrypted_url)

        # Extract the file to a specific location
        extract_directory = WEIGHTS_PATH
        extract_zip_file(DOWNLOAD_PTS_FILE_PATH, extract_directory)

    except Exception as e:
        print(f"Failed to decrypt or download: {str(e)}")
