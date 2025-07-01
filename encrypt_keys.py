from cryptography.fernet import Fernet

key = Fernet.generate_key()
with open("fernet.key", "wb") as f:
    f.write(key)

import json

# Load the key
with open("fernet.key", "rb") as f:
    key = f.read()

fernet = Fernet(key)

# Your secrets
secrets = {
    "LLAMA_CLOUD_API_KEY": "llx-DAetMDYuEx3KwNWM4eRyhGyQMA8cSQOXP5KTSMrwtFWnq6a4",
    "OPENAI_API_KEY": "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"
}

# Encrypt and save
token = fernet.encrypt(json.dumps(secrets).encode())
with open("encrypted_keys.enc", "wb") as f:
    f.write(token)

