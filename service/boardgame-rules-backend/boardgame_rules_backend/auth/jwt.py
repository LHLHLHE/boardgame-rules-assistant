from jose import JWTError, jwt

from boardgame_rules_backend.settings import app_config


def encode_token(payload: dict) -> str:
    return jwt.encode(
        payload,
        app_config.jwt_secret,
        algorithm=app_config.jwt_algorithm,
    )


def decode_token(token: str) -> dict:
    return jwt.decode(
        token,
        app_config.jwt_secret,
        algorithms=[app_config.jwt_algorithm],
    )


def user_id_from_payload(payload: dict) -> int:
    sub = payload.get("sub")
    if sub is None:
        raise JWTError("missing sub")
    return int(sub)
