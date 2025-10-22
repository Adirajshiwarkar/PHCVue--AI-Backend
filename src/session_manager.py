import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)

class SessionState(BaseModel):
    """Holds the complete state of a patient consultation."""
    session_id: str
    language: str
    patient_id: Optional[str] = None
    patient_info: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []
    status: str = "starting"
    translated_history: Optional[str] = None

class SessionManager:
    """
    Manages state and tasks for a single, interruptible voice session.
    """
    def __init__(self, session_id: str, language_code: str):
        self.state = SessionState(session_id=session_id, language=language_code)
        self.transcript_queue = asyncio.Queue()
        self.last_ai_response: Optional[str] = None
        
        # --- Advanced Voice Management Attributes ---
        self.is_speaking: bool = False
        self.websocket_active: bool = True
        
        # ID to track the current generation of AI response
        self.generation_id: int = 0
        
        # Buffer for incoming transcripts
        self.transcript_buffer: List[str] = []
        
        # References to all currently running tasks that can be cancelled
        self.current_processing_task: Optional[asyncio.Task] = None
        self.current_sentence_tasks: List[asyncio.Task] = []

        logger.info(f"[{self.state.session_id}] 🆕 Session created for language '{language_code}'.")

    def load_patient_data(self, patient_record: dict, translated_history: str):
        """Loads existing patient data into the session state."""
        self.state.patient_id = patient_record.get("patient_id")
        self.state.patient_info = patient_record
        self.state.translated_history = translated_history
        self.state.conversation_history.append({
            "role": "system",
            "content": f"The patient's known medical background is: {translated_history}"
        })
        logger.info(f"[{self.state.session_id}] 📂 Patient data loaded for ID: {self.state.patient_id}.")

    async def interrupt(self):
        """
        Safely interrupts all ongoing AI processing (thinking) and TTS (speaking) 
        tasks for this session. This is the core of the barge-in logic.
        """
        session_id = self.state.session_id

        # ✅ FIX: The condition is now more robust. It checks if the AI is
        # speaking OR if it's processing in the background. This allows
        # interrupting the AI even before it starts talking.
        is_processing = self.current_processing_task and not self.current_processing_task.done()
        if not self.is_speaking and not is_processing:
            return # Nothing to interrupt

        logger.info(f"[{session_id}] 🔴 BARGE-IN: Interrupt triggered.")
        
        self.generation_id += 1
        logger.info(f"[{session_id}] 🔄 Generation ID incremented to: {self.generation_id}.")
        
        tasks_to_cancel = []
        if self.current_processing_task and not self.current_processing_task.done():
            tasks_to_cancel.append(self.current_processing_task)
        
        tasks_to_cancel.extend([task for task in self.current_sentence_tasks if not task.done()])
        
        if tasks_to_cancel:
            logger.info(f"[{session_id}] 🛑 Identifying {len(tasks_to_cancel)} active tasks for cancellation...")
            for task in tasks_to_cancel:
                task.cancel()
            
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            logger.info(f"[{session_id}] ✅ All active tasks successfully cancelled.")
        else:
            logger.info(f"[{session_id}] No active tasks needed cancellation.")

        # Reset all state variables after cancellation
        self.is_speaking = False
        self.current_processing_task = None
        self.current_sentence_tasks.clear()
        self.transcript_buffer.clear()
        logger.info(f"[{session_id}] ✅ Interrupt complete.")