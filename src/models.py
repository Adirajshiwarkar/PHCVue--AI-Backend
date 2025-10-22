from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import asyncio

class ConsultationState(BaseModel):
    """Holds the complete state of a patient consultation."""
    session_id: str
    patient_id: Optional[str] = None # Will be filled in after identification
    language: str
    patient_info: Dict = Field(default_factory=dict)
    translated_history: str = ""
    conversation_history: List[Dict] = Field(default_factory=list)
    status: str = "identifying" # identifying -> active -> awaiting_finalization -> complete