from fastapi import Header, HTTPException
from typing import Optional, Dict
from utils.logger import get_logger

logger = get_logger(__name__)

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """Simulates decoding a JWT token from the Authorization header."""
    logger.info("AUTH: Validating token...")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    
    token = authorization.split(" ")[1]
    if token == "patient123":
        logger.info("AUTH: Authenticated as Patient 'PAT-001'.")
        return {"user_id": "PAT-001", "role": "patient"}
    elif token == "doctor456":
        logger.info("AUTH: Authenticated as Doctor 'DOC-001'.")
        return {"user_id": "DOC-001", "role": "doctor"}
    else:
        raise HTTPException(status_code=401, detail="Invalid token.")