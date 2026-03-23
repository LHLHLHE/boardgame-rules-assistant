import datetime as dt

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    id: int
    username: str
    email: str | None = None
    is_admin: bool
    is_staff: bool
    created_at: dt.datetime

    model_config = {"from_attributes": True}


class AuthUser(BaseModel):
    id: int
    username: str
    email: str | None = None
    is_admin: bool
    is_staff: bool
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = {"from_attributes": True}
