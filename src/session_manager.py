# import asyncio
# from typing import List, Dict, Any, Optional
# from pydantic import BaseModel
# from fastapi import WebSocket
# from utils.logger import get_logger

# logger = get_logger(__name__)

# class SessionState(BaseModel):
#     """Holds the complete state of a patient consultation."""
#     session_id: str
#     language: str
#     patient_id: Optional[str] = None
#     patient_info: Dict[str, Any] = {}
#     conversation_history: List[Dict[str, str]] = []
#     status: str = "starting"
#     translated_history: Optional[str] = None

# class SessionManager:
#     """
#     Manages state and tasks for a single, interruptible voice session.
#     ✅ FIXED: Now uses exact logic from working second codebase
#     """
#     def __init__(self, session_id: str, language_code: str):
#         self.state = SessionState(session_id=session_id, language=language_code)
#         self.transcript_queue = asyncio.Queue()
#         self.last_ai_response: Optional[str] = None
        
#         # --- ✅ FIX: Store websocket reference (from second code) ---
#         self._websocket: Optional[WebSocket] = None
#         self._websocket_id: Optional[int] = None
        
#         # --- Advanced Voice Management Attributes ---
#         self.is_speaking: bool = False
#         self.websocket_active: bool = True
        
#         # ✅ FIX: Separate generation IDs like second code
#         self.generation_id: int = 0  # For processing tasks
#         self.interrupt_generation_id: int = 0  # For interrupts (from second code)
        
#         # Buffer for incoming transcripts
#         self.transcript_buffer: List[str] = []
        
#         # References to all currently running tasks that can be cancelled
#         self.current_processing_task: Optional[asyncio.Task] = None
#         self.current_sentence_tasks: List[asyncio.Task] = []
        
#         # ✅ FIX: Add cancel event (from second code)
#         self.cancel_event = asyncio.Event()
        
#         # ✅ FIX: Add interrupt cooldown (from second code)
#         self.last_interrupt_time = None
#         self.interrupt_cooldown_seconds = 1.0

#         logger.info(f"[{self.state.session_id}] 🆕 Session created for language '{language_code}'.")

#     def set_websocket(self, websocket: WebSocket):
#         """✅ FIX: Bind WebSocket to this session (from second code)."""
#         self._websocket = websocket
#         self._websocket_id = id(websocket)
#         logger.info(f"[{self.state.session_id}] 🔗 WebSocket bound (ID: {self._websocket_id})")
    
#     def validate_websocket(self, websocket: WebSocket) -> bool:
#         """✅ FIX: Verify WebSocket belongs to this session (from second code)."""
#         is_valid = self._websocket_id == id(websocket)
#         if not is_valid:
#             logger.error(
#                 f"[{self.state.session_id}] ❌ WebSocket mismatch! "
#                 f"Expected: {self._websocket_id}, Got: {id(websocket)}"
#             )
#         return is_valid

#     def load_patient_data(self, patient_record: dict, translated_history: str):
#         """Loads existing patient data into the session state."""
#         self.state.patient_id = patient_record.get("patient_id")
#         self.state.patient_info = patient_record
#         self.state.translated_history = translated_history
#         self.state.conversation_history.append({
#             "role": "system",
#             "content": f"The patient's known medical background is: {translated_history}"
#         })
#         logger.info(f"[{self.state.session_id}] 📂 Patient data loaded for ID: {self.state.patient_id}.")

#     async def interrupt(self):
#         """
#         ✅ FIX: Complete interrupt logic from second codebase
#         Safely interrupts all ongoing AI processing and TTS tasks.
#         """
#         try:
#             session_id = self.state.session_id
            
#             # ✅ FIX: Validate websocket ownership (from second code)
#             if self._websocket and not self.validate_websocket(self._websocket):
#                 logger.error(f"[{session_id}] ❌ CRITICAL: Interrupt called with WRONG WebSocket!")
#                 return
            
#             logger.info(f"[{session_id}] 🔴 BARGE-IN: Interrupt triggered")
            
#             # ✅ FIX: Send CLEAR to frontend (from second code)
#             if self._websocket:
#                 try:
#                     await self._websocket.send_json({"type": "clear"})
#                     logger.info(f"[{session_id}] ✅ CLEAR sent to frontend")
#                 except Exception as e:
#                     logger.error(f"[{session_id}] ❌ Failed to send CLEAR: {e}")
#                     return
            
#             # ✅ FIX: Increment interrupt_generation_id (from second code)
#             self.interrupt_generation_id += 1
#             current_gen = self.interrupt_generation_id
#             logger.info(f"[{session_id}] 🔄 Interrupt Generation ID: {current_gen}")
            
#             # ✅ FIX: Set cancel event (from second code)
#             self.cancel_event.set()
            
#             # Check if was speaking
#             was_speaking = self.is_speaking
#             self.is_speaking = False
            
#             if not was_speaking:
#                 is_processing = self.current_processing_task and not self.current_processing_task.done()
#                 if not is_processing:
#                     logger.info(f"[{session_id}] ℹ️ Not speaking/processing, interrupt ignored")
#                     self.cancel_event.clear()
#                     return
            
#             # ✅ FIX: Build task list with validation (from second code)
#             tasks_to_cancel = []
            
#             if self.current_processing_task and not self.current_processing_task.done():
#                 task_name = self.current_processing_task.get_name()
#                 if session_id in task_name or not task_name:
#                     tasks_to_cancel.append(("Processing", self.current_processing_task))
#                     logger.info(f"[{session_id}] Will cancel Processing task: {task_name}")
            
#             for i, task in enumerate(self.current_sentence_tasks):
#                 if not task.done():
#                     task_name = task.get_name()
#                     if session_id in task_name or not task_name:
#                         tasks_to_cancel.append((f"Sentence-{i}", task))
#                         logger.info(f"[{session_id}] Will cancel Sentence-{i} task")
            
#             # Cancel tasks
#             for name, task in tasks_to_cancel:
#                 logger.info(f"[{session_id}] 🛑 Cancelling {name} task")
#                 task.cancel()
            
#             if tasks_to_cancel:
#                 try:
#                     await asyncio.wait_for(
#                         asyncio.gather(*[task for _, task in tasks_to_cancel], return_exceptions=True),
#                         timeout=1.0
#                     )
#                     logger.info(f"[{session_id}] ✅ All tasks cancelled")
#                 except asyncio.TimeoutError:
#                     logger.warning(f"[{session_id}] ⚠️ Task cancellation timeout")
            
#             # ✅ FIX: Clear transcript buffer (from second code)
#             if self.transcript_buffer:
#                 logger.info(f"[{session_id}] 🧹 Clearing {len(self.transcript_buffer)} buffered transcripts")
#                 self.transcript_buffer.clear()
            
#             # Reset state
#             self.current_processing_task = None
#             self.current_sentence_tasks.clear()
#             self.cancel_event.clear()
            
#             logger.info(f"[{session_id}] ✅ INTERRUPT COMPLETE")
        
#         except Exception as e:
#             # logger.error(f"[{self.state.session_id}] ❌ Error in interrupt: {e}", exc_info=True)

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import WebSocket
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
    ✅ FIXED: Now uses exact logic from working second codebase
    """
    def __init__(self, session_id: str, language_code: str):
        self.state = SessionState(session_id=session_id, language=language_code)
        self.transcript_queue = asyncio.Queue()
        self.last_ai_response: Optional[str] = None
        
        # --- ✅ FIX: Store websocket reference (from second code) ---
        self._websocket: Optional[WebSocket] = None
        self._websocket_id: Optional[int] = None
        
        # --- Advanced Voice Management Attributes ---
        self.is_speaking: bool = False
        self.websocket_active: bool = True
        
        # ✅ FIX: Separate generation IDs like second code
        self.generation_id: int = 0  # For processing tasks
        self.interrupt_generation_id: int = 0  # For interrupts (from second code)
        
        # Buffer for incoming transcripts
        self.transcript_buffer: List[str] = []
        
        # References to all currently running tasks that can be cancelled
        self.current_processing_task: Optional[asyncio.Task] = None
        self.current_sentence_tasks: List[asyncio.Task] = []
        
        # ✅ FIX: Add cancel event (from second code)
        self.cancel_event = asyncio.Event()
        
        # ✅ FIX: Add interrupt cooldown (from second code)
        self.last_interrupt_time = None
        self.interrupt_cooldown_seconds = 1.0

        logger.info(f"[{self.state.session_id}] 🆕 Session created for language '{language_code}'.")

    def set_websocket(self, websocket: WebSocket):
        """✅ FIX: Bind WebSocket to this session (from second code)."""
        self._websocket = websocket
        self._websocket_id = id(websocket)
        logger.info(f"[{self.state.session_id}] 🔗 WebSocket bound (ID: {self._websocket_id})")
    
    def validate_websocket(self, websocket: WebSocket) -> bool:
        """✅ FIX: Verify WebSocket belongs to this session (from second code)."""
        is_valid = self._websocket_id == id(websocket)
        if not is_valid:
            logger.error(
                f"[{self.state.session_id}] ❌ WebSocket mismatch! "
                f"Expected: {self._websocket_id}, Got: {id(websocket)}"
            )
        return is_valid

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
        ✅ FIX: Complete interrupt logic from second codebase
        Safely interrupts all ongoing AI processing and TTS tasks.
        """
        try:
            session_id = self.state.session_id
            
            # ✅ FIX: Validate websocket ownership (from second code)
            if self._websocket and not self.validate_websocket(self._websocket):
                logger.error(f"[{session_id}] ❌ CRITICAL: Interrupt called with WRONG WebSocket!")
                return
            
            logger.info(f"[{session_id}] 🔴 BARGE-IN: Interrupt triggered")
            
            # ✅ FIX: Send CLEAR to frontend (from second code)
            if self._websocket:
                try:
                    await self._websocket.send_json({"type": "clear"})
                    logger.info(f"[{session_id}] ✅ CLEAR sent to frontend")
                except Exception as e:
                    logger.error(f"[{session_id}] ❌ Failed to send CLEAR: {e}")
                    return
            
            # ✅ FIX: Increment interrupt_generation_id (from second code)
            self.interrupt_generation_id += 1
            current_gen = self.interrupt_generation_id
            logger.info(f"[{session_id}] 🔄 Interrupt Generation ID: {current_gen}")
            
            # ✅ FIX: Set cancel event (from second code)
            self.cancel_event.set()
            
            # Check if was speaking
            was_speaking = self.is_speaking
            self.is_speaking = False
            
            if not was_speaking:
                is_processing = self.current_processing_task and not self.current_processing_task.done()
                if not is_processing:
                    logger.info(f"[{session_id}] ℹ️ Not speaking/processing, interrupt ignored")
                    self.cancel_event.clear()
                    return
            
            # ✅ FIX: Build task list with validation (from second code)
            tasks_to_cancel = []
            
            if self.current_processing_task and not self.current_processing_task.done():
                task_name = self.current_processing_task.get_name()
                if session_id in task_name or not task_name:
                    tasks_to_cancel.append(("Processing", self.current_processing_task))
                    logger.info(f"[{session_id}] Will cancel Processing task: {task_name}")
            
            for i, task in enumerate(self.current_sentence_tasks):
                if not task.done():
                    task_name = task.get_name()
                    if session_id in task_name or not task_name:
                        tasks_to_cancel.append((f"Sentence-{i}", task))
                        logger.info(f"[{session_id}] Will cancel Sentence-{i} task")
            
            # Cancel tasks
            for name, task in tasks_to_cancel:
                logger.info(f"[{session_id}] 🛑 Cancelling {name} task")
                task.cancel()
            
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*[task for _, task in tasks_to_cancel], return_exceptions=True),
                        timeout=1.0
                    )
                    logger.info(f"[{session_id}] ✅ All tasks cancelled")
                except asyncio.TimeoutError:
                    logger.warning(f"[{session_id}] ⚠️ Task cancellation timeout")
            
            # ✅ FIX: Clear transcript buffer (from second code)
            if self.transcript_buffer:
                logger.info(f"[{session_id}] 🧹 Clearing {len(self.transcript_buffer)} buffered transcripts")
                self.transcript_buffer.clear()
            
            # Reset state
            self.current_processing_task = None
            self.current_sentence_tasks.clear()
            self.cancel_event.clear()
            
            logger.info(f"[{session_id}] ✅ INTERRUPT COMPLETE")
        
        except Exception as e:
            logger.error(f"[{self.state.session_id}] ❌ Error in interrupt: {e}", exc_info=True)
