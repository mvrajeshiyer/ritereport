import json
from cryptography.fernet import Fernet

def load_secrets(enc_file='utils/encrypted_keys.enc', key_file='fernet.key'):
    """
    Decrypts and loads API keys from an encrypted file using a Fernet key from a file.
    """
    with open(key_file, "rb") as kf:
        secret_key = kf.read()
    fernet = Fernet(secret_key)
    with open(enc_file, 'rb') as f:
        encrypted = f.read()
    decrypted = fernet.decrypt(encrypted)
    return json.loads(decrypted.decode())
