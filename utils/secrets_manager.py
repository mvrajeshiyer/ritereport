import os
from cryptography.fernet import Fernet

def load_encrypted_keys(file_path):
    with open(file_path, 'rb') as file:
        encrypted_data = file.read()
    return encrypted_data

def decrypt_key(encrypted_key, secret_key):
    fernet = Fernet(secret_key)
    decrypted_key = fernet.decrypt(encrypted_key)
    return decrypted_key.decode()

def get_api_keys():
    secret_key = os.environ.get("SECRET_KEY")  # Ensure the secret key is set in the environment
    encrypted_keys = load_encrypted_keys('../encrypted_keys.enc')
    api_keys = {}
    
    # Assuming the encrypted_keys.enc contains multiple keys in a specific format
    # For example, it could be a JSON string of encrypted keys
    encrypted_keys_dict = json.loads(decrypt_key(encrypted_keys, secret_key))
    
    api_keys['LLAMA_CLOUD_API_KEY'] = decrypt_key(encrypted_keys_dict['LLAMA_CLOUD_API_KEY'], secret_key)
    api_keys['OPENAI_API_KEY'] = decrypt_key(encrypted_keys_dict['OPENAI_API_KEY'], secret_key)
    
    return api_keys