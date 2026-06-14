"""Symmetric encryption for secrets at rest (BYOK LLM keys).

Uses Fernet (AES-128-CBC + HMAC authenticated encryption). The key comes from
the CODEATLAS_SECRET_KEY env var if set; otherwise a key is generated once and
persisted to data/secret.key (gitignored, mode 0600) so encrypted data survives
restarts.
"""

import os
from functools import lru_cache

from cryptography.fernet import Fernet

from ..config import DATA_DIR

SECRET_FILE = DATA_DIR / "secret.key"


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    env_key = os.environ.get("CODEATLAS_SECRET_KEY")
    if env_key:
        return Fernet(env_key.encode())

    if SECRET_FILE.exists():
        return Fernet(SECRET_FILE.read_bytes())

    key = Fernet.generate_key()
    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRET_FILE.write_bytes(key)
    try:
        os.chmod(SECRET_FILE, 0o600)
    except OSError:
        pass
    return Fernet(key)


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    return _fernet().decrypt(token.encode()).decode()
