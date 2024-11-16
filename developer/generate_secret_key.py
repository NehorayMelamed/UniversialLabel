from cryptography.fernet import Fernet
import base64
import hashlib
from common.general_parameters import SECRET_KEY_PATH, ENCRYPTED_UTL_PATH

def generate_strong_secret_key():
    return Fernet.generate_key().decode()

def derive_encryption_key(secret_key):
    return base64.urlsafe_b64encode(hashlib.sha256(secret_key.encode()).digest())

def encrypt_url(secret_key, url):
    encryption_key = derive_encryption_key(secret_key)
    fernet = Fernet(encryption_key)
    encrypted_url = fernet.encrypt(url.encode())
    return encrypted_url.decode()

def save_to_file(data, file_path):
    with open(file_path, 'w') as file:
        file.write(data)

if __name__ == "__main__":
    # Generate a strong secret key
    secret_key = generate_strong_secret_key()
    print(f"Generated Secret Key: {secret_key}")

    # Save the secret key to a file
    save_to_file(secret_key, SECRET_KEY_PATH)

    # URL to be encrypted (replace this with your Mega link)
    original_url = "https://mega.nz/file/wz0AQZbD#M2sterOBiXOiUkAVf2AiOF2YAUW3N6FWdXLlFEx_kpU"

    # Encrypt the URL
    encrypted_url = encrypt_url(secret_key, original_url)
    print(f"Encrypted URL: {encrypted_url}")

    # Save the encrypted URL to a file
    save_to_file(encrypted_url, ENCRYPTED_UTL_PATH)
