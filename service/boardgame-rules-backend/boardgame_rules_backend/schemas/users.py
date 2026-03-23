import datetime as dt

from pydantic import BaseModel, Field, model_validator


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8)
    email: str | None = None
    is_admin: bool = False
    is_staff: bool = False


class UserUpdate(BaseModel):
    password: str | None = Field(None, min_length=8)
    email: str | None = None
    is_admin: bool | None = None
    is_staff: bool | None = None

    @model_validator(mode="after")
    def reject_explicit_null_password(self):
        if "password" in self.model_fields_set and self.password is None:
            msg = "password cannot be null"
            raise ValueError(msg)
        return self


class UserRead(BaseModel):
    id: int
    username: str
    email: str | None
    is_admin: bool
    is_staff: bool
    created_at: dt.datetime
    updated_at: dt.datetime

    model_config = {"from_attributes": True}
