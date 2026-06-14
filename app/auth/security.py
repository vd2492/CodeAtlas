"""Password hashing helpers (stdlib only — no external crypto dependency).

Uses PBKDF2-HMAC-SHA256 with a per-password salt. Good enough for an
internal tool; swap for argon2/bcrypt later if desired.
"""

import hashlib
import hmac
import os

_ITERATIONS = 240_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters, salt_hex, digest_hex = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(expected, actual)
    except (ValueError, AttributeError):
        return False
