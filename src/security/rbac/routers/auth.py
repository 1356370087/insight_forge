"""``/auth/*`` router: registration, verification, login, refresh, sessions, password flows."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_transactional_db
from ..dependencies import get_current_principal
from ..principal import Principal
from ..repositories import collect_user_permissions, get_session, get_user_by_id
from ..schemas import (
    AcceptedResponse,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    LogoutAllResponse,
    LogoutResponse,
    RefreshRequest,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    SessionItem,
    TokenPair,
    UserMe,
    VerifyEmailRequest,
)
from ..services import auth as auth_service
from ..services import sessions as sessions_service
from ..services.auth import AuthError
from ..settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _no_store(body: dict, status_code: int = status.HTTP_200_OK) -> JSONResponse:
    """Wrap a token-bearing response with ``Cache-Control: no-store``."""
    return JSONResponse(status_code=status_code, content=body, headers={"Cache-Control": "no-store"})


def _raise_auth_error(exc: AuthError) -> None:
    """Map an :class:`AuthError` to an ``HTTPException``."""
    raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


def _client_identity(request: Request) -> str:
    """Return a stable identity for rate limiting (IP, falling back to UA)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _request_meta(request: Request) -> tuple[str, str | None]:
    """Return ``(ip_address, user_agent)`` from the request."""
    ip = _client_identity(request)
    return ip, request.headers.get("user-agent")


def _token_pair(pair) -> TokenPair:
    """Convert a service :class:`TokenPairRecord` to the API model."""
    return TokenPair(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.access_expires_in,
        refresh_expires_in=pair.refresh_expires_in,
        session_id=pair.session_id,
    )


@router.post("/register", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
async def register(
    request_body: RegisterRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_transactional_db),
) -> AcceptedResponse:
    """Register a new user and email a verification link (always 202)."""
    settings = get_settings()
    if not settings.open_registration:
        raise HTTPException(status_code=403, detail="registration_closed")
    ip, _ = _request_meta(http_request)
    try:
        await auth_service.register(
            db,
            email=request_body.email,
            password=request_body.password,
            display_name=request_body.display_name,
            settings=settings,
            identity_for_rate_limit=ip,
        )
    except AuthError as exc:
        await db.commit()
        _raise_auth_error(exc)
    return AcceptedResponse(message="If the email is available, a verification link has been sent.")


@router.post("/verify-email", response_model=UserMe)
async def verify_email(
    request_body: VerifyEmailRequest,
    db: AsyncSession = Depends(get_transactional_db),
) -> UserMe:
    """Verify an email address with a one-time token."""
    settings = get_settings()
    try:
        user = await auth_service.verify_email(db, token=request_body.token, settings=settings)
    except AuthError as exc:
        await db.commit()
        _raise_auth_error(exc)
    role_codes, _ = await collect_user_permissions(db, str(user.id))
    return UserMe(
        id=str(user.id), email=user.email, display_name=user.display_name,
        status=user.status, roles=sorted(role_codes), permissions=[], authz_version=int(user.authz_version),
    )


@router.post(
    "/email-verification/resend", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse,
)
async def resend_verification(
    request_body: ResendVerificationRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_transactional_db),
) -> AcceptedResponse:
    """Re-send the verification email (always 202)."""
    settings = get_settings()
    ip, _ = _request_meta(http_request)
    try:
        await auth_service.resend_verification(
            db, email=request_body.email, settings=settings, identity_for_rate_limit=ip,
        )
    except AuthError as exc:
        await db.commit()
        _raise_auth_error(exc)
    return AcceptedResponse(message="If the email is pending verification, a new link has been sent.")


@router.post("/login", response_model=TokenPair)
async def login(
    request_body: LoginRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_transactional_db),
) -> JSONResponse:
    """Authenticate and return a token pair (``Cache-Control: no-store``)."""
    settings = get_settings()
    ip, user_agent = _request_meta(http_request)
    try:
        pair = await auth_service.login(
            db,
            email=request_body.email,
            password=request_body.password,
            settings=settings,
            user_agent=user_agent,
            ip_address=ip,
        )
    except AuthError as exc:
        # Failed-login counters and rate-limit buckets are security state and
        # must survive the HTTP error response.
        await db.commit()
        _raise_auth_error(exc)
    return _no_store(_token_pair(pair).model_dump(mode="json"))


@router.post("/refresh", response_model=TokenPair)
async def refresh(
    request_body: RefreshRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_transactional_db),
) -> JSONResponse:
    """Rotate a refresh token and return a fresh pair (``no-store``)."""
    settings = get_settings()
    ip, user_agent = _request_meta(http_request)
    try:
        pair = await auth_service.refresh(
            db, refresh_token=request_body.refresh_token, settings=settings,
            ip_address=ip, user_agent=user_agent,
        )
    except AuthError as exc:
        # Reuse detection revokes the session family before raising.
        await db.commit()
        _raise_auth_error(exc)
    return _no_store(_token_pair(pair).model_dump(mode="json"))


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_transactional_db),
) -> LogoutResponse:
    """Revoke the current session."""
    if principal.session_id is None:
        return LogoutResponse(revoked=False)
    await auth_service.logout(db, session_id=principal.session_id, user_id=principal.user_id)
    return LogoutResponse(revoked=True)


@router.post("/logout-all", response_model=LogoutAllResponse)
async def logout_all(
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_transactional_db),
) -> LogoutAllResponse:
    """Revoke every session for the current user (except optionally the current one)."""
    count = await auth_service.logout_all(
        db, user_id=principal.user_id, except_session_id=principal.session_id,
    )
    return LogoutAllResponse(revoked_count=count)


@router.get("/me", response_model=UserMe)
async def me(
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_transactional_db),
) -> UserMe:
    """Return the caller's profile and effective RBAC.

    Pending-approval users may reach this endpoint (and the pending page) but
    not the research APIs.
    """
    role_codes, permission_codes = await collect_user_permissions(db, principal.user_id)
    user = await get_user_by_id(db, principal.user_id)
    display_name = user.display_name if user is not None else None
    return UserMe(
        id=principal.user_id,
        email=principal.email,
        display_name=display_name,
        status=principal.status,
        roles=sorted(role_codes),
        permissions=sorted(permission_codes),
        authz_version=principal.authz_version,
        session_id=principal.session_id,
    )


@router.get("/sessions", response_model=list[SessionItem])
async def list_sessions(
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_transactional_db),
) -> list[SessionItem]:
    """List the caller's device sessions."""
    sessions = await sessions_service.list_user_sessions(db, principal.user_id)
    return [
        SessionItem(
            id=str(session.id),
            created_at=session.created_at.isoformat(),
            last_activity_at=session.last_activity_at.isoformat(),
            user_agent=session.user_agent,
            ip_address=session.ip_address,
            is_current=(str(session.id) == principal.session_id),
            is_revoked=session.is_revoked,
        )
        for session in sessions
    ]


@router.delete("/sessions/{session_id}", response_model=LogoutResponse)
async def delete_session(
    session_id: str,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_transactional_db),
) -> LogoutResponse:
    """Revoke one of the caller's own sessions by id."""
    target = await get_session(db, session_id)
    if target is None or str(target.user_id) != principal.user_id:
        raise HTTPException(status_code=404, detail="Session not found")
    revoked = await sessions_service.revoke_session(
        db, session_id, reason=sessions_service.SessionRevocationReason.LOGOUT, actor_id=principal.user_id,
    )
    return LogoutResponse(revoked=revoked)


@router.post(
    "/password/forgot", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse,
)
async def forgot_password(
    request_body: ForgotPasswordRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_transactional_db),
) -> AcceptedResponse:
    """Email a password-reset link if the account exists (always 202)."""
    settings = get_settings()
    ip, _ = _request_meta(http_request)
    try:
        await auth_service.forgot_password(
            db, email=request_body.email, settings=settings, identity_for_rate_limit=ip,
        )
    except AuthError as exc:
        await db.commit()
        _raise_auth_error(exc)
    return AcceptedResponse(message="If the email matches an account, a reset link has been sent.")


@router.post("/password/reset", response_model=AcceptedResponse)
async def reset_password(
    request_body: ResetPasswordRequest,
    db: AsyncSession = Depends(get_transactional_db),
) -> AcceptedResponse:
    """Reset a password with a one-time token."""
    settings = get_settings()
    try:
        await auth_service.reset_password(
            db, token=request_body.token, new_password=request_body.password, settings=settings,
        )
    except AuthError as exc:
        await db.commit()
        _raise_auth_error(exc)
    return AcceptedResponse(message="Password reset. You can now sign in.")


@router.post("/password/change", response_model=AcceptedResponse)
async def change_password(
    request_body: ChangePasswordRequest,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_transactional_db),
) -> AcceptedResponse:
    """Change the password after verifying the current one (revokes sessions)."""
    settings = get_settings()
    try:
        await auth_service.change_password(
            db,
            user_id=principal.user_id,
            current_password=request_body.current_password,
            new_password=request_body.new_password,
            settings=settings,
        )
    except AuthError as exc:
        _raise_auth_error(exc)
    return AcceptedResponse(message="Password changed. Please sign in again.")
