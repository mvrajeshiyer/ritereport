import os
import json
from cryptography.fernet import Fernet

export SECRET_KEY="O8lc6uPUdNdksa9DnnfaF6Xv-McvCsGwq_RbrpfDF2w="

def load_secrets(enc_file='encrypted_keys.enc', key_env_var='SECRET_KEY'):
    """
    Decrypts and loads API keys from an encrypted file using a Fernet key from an environment variable.
    """
    secret_key = os.environ.get(key_env_var)
    if not secret_key:
        raise ValueError("SECRET_KEY environment variable not set.")
    fernet = Fernet(secret_key)
    with open(enc_file, 'rb') as f:
        encrypted = f.read()
    decrypted = fernet.decrypt(encrypted)
    return json.loads(decrypted.decode())
