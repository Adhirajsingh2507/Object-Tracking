"""
auth.py — JWT + bcrypt Authentication
----------------------------------------
Token-based auth for API routes.
Uses bcrypt directly (not passlib) for compatibility with bcrypt>=4.0.
"""

import datetime

import bcrypt
import jwt
from fastapi import HTTPException, Request

from backend.database import get_user_by_username, get_user_by_id, create_user

# ── Config ──
SECRET_KEY = "tracker-secret-key-change-in-production"
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24


# ── Password helpers ──
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT helpers ──
def create_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Auth dependency ──
def get_current_user(request: Request) -> dict:
    """FastAPI dependency: extract and validate JWT from Authorization header or query param."""
    token = None

    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]

    # Fallback: check query param (for MJPEG stream in <img> tag)
    if not token:
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(token)
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── Registration / Login ──
def register_user(username: str, email: str, password: str) -> dict:
    existing = get_user_by_username(username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed = hash_password(password)
    user_id = create_user(username, email, hashed)
    token = create_token(user_id, username)
    return {"token": token, "user_id": user_id, "username": username}


def login_user(username: str, password: str) -> dict:
    user = get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"], user["username"])
    return {"token": token, "user_id": user["id"], "username": user["username"]}
