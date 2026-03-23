import bcrypt


def hash_password_bcrypt(plain: str) -> str:
    """Bcrypt hash (compatible with hashes produced by passlib bcrypt)."""
    pw = plain.encode("utf-8")
    if len(pw) > 72:
        raise ValueError("Password must be at most 72 bytes in UTF-8")
    digest = bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12))
    return digest.decode("ascii")


def verify_password_bcrypt(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False
