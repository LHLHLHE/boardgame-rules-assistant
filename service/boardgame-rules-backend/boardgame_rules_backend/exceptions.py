class GameNotFound(Exception):
    detail = "Game not found"


class DuplicateGameTitle(Exception):
    """Another game with the same title (case-insensitive) already exists."""

    detail = "A game with this title already exists"


class EmptyFileError(Exception):
    detail = "Empty file"


class RulesProcessingInProgress(Exception):
    """Rules document is pending or processing; upload would replace an in-flight job."""

    detail = (
        "Rules are queued or being indexed. Wait until status is indexed or failed, "
        "then upload again."
    )


class AuthServiceError(Exception):
    status_code: int = 400
    detail: str = "Error"


class AuthInvalidCredentials(AuthServiceError):
    status_code = 401
    detail = "Invalid username or password"


class AuthPanelAccessDenied(AuthServiceError):
    status_code = 403
    detail = "User is not allowed to access the admin panel"


class UsernameAlreadyExists(AuthServiceError):
    status_code = 409
    detail = "Username already exists"


class AdminUserNotFound(AuthServiceError):
    status_code = 404
    detail = "User not found"


class LastAdminRemovalError(AuthServiceError):
    status_code = 400
    detail = "Cannot remove or delete the last admin"


class InitialAdminAlreadyExistsError(AuthServiceError):
    status_code = 409
    detail = "An admin user already exists. Use --force to create another."


class InvalidOrExpiredTokenError(AuthServiceError):
    status_code = 401
    detail = "Invalid or expired token"


class TokenSubjectUserMissingError(AuthServiceError):
    status_code = 401
    detail = "User not found"
