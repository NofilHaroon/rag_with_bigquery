"""
auth.py

API key authentication for FastAPI endpoints.
"""

from fastapi import HTTPException, status, Depends, Header
from typing import Optional
import logging

from utils import get_api_keys

logger = logging.getLogger(__name__)


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Verify API key from X-API-Key header against configured keys."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )
    
    valid_keys = get_api_keys()
    
    if not valid_keys:
        logger.warning("No API keys configured. Allowing all requests.")
        return x_api_key
    
    if x_api_key not in valid_keys:
        logger.warning(f"Invalid API key attempted: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return x_api_key
