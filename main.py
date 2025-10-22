# main.py (Updated with corrected idle prompt logic)

import asyncio
import os
import uuid
import json
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# --- Import Custom Project Modules ---
from src.session_manager import SessionManager
from src.triage_engine import TriageEngine
from src.services.transcription_service import TranscriptionService
from utils.database import get_db, close_db, find_closest_patient_by_name, save_consultation_details
from utils.external_apis import generate_tts_stream, call_llm, translate_text
from utils.logger import get_logger

# --- Initialize Logger and Application ---
logger = get_logger(__name__)
app = FastAPI(title="PHCVue AI Healthcare Assistant")
triage_engine = TriageEngine()
active_sessions: dict[str, SessionManager] = {}

# --- Add CORS Middleware & Lifecycle Events ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
@app.on_event("shutdown")
async def shutdown_event():
    await close_db()

SUPPORTED_LANGUAGES = ["en-US", "en-IN", "hi-IN", "mr-IN", "ta-IN", "te-IN", "kn-IN"]
ENGLISH_CODES = ["en-US", "en-IN"]

# ==============================================================================
# REFACTORED CONVERSATION HANDLER CLASS
# ==============================================================================
class ConversationHandler:
    """Encapsulates all logic for handling a single WebSocket conversation."""
    def __init__(self, websocket: WebSocket, session: SessionManager):
        self.websocket = websocket
        self.session = session

    async def run(self):
        """Main entry point to start the conversation flow."""
        transcription_service = TranscriptionService(self.session, self.websocket)
        initial_greeting = await translate_text(
            "Hello, I am your AI health assistant. Please tell me your full name to begin.", 
            self.session.state.language
        )
        await self._stream_tts_response(initial_greeting)
        
        await asyncio.gather(
            self._listen_for_transcripts(), 
            transcription_service.start_stream()
        )

    # --- ✅ THIS IS THE FIX ---
    async def _listen_for_transcripts(self):
        """
        Waits for user transcripts and orchestrates the AI response with robust
        idle timeout logic.
        """
        idle_seconds_counter = 0
        SILENCE_THRESHOLD = 15  # Fire prompt after 15 seconds of true silence

        while self.session.websocket_active and self.session.state.status != "awaiting_finalization":
            try:
                # Wait for a transcript, but with a short 1-second timeout.
                transcript = await asyncio.wait_for(self.session.transcript_queue.get(), timeout=1.0)

                # If we receive a transcript, user is active. Reset counter.
                idle_seconds_counter = 0

                # Interrupt AI if it's speaking.
                if self.session.is_speaking:
                    await self.session.interrupt()
                
                # Process the user's speech in the background.
                gen_id = self.session.generation_id
                processing_task = asyncio.create_task(self._process_transcript(transcript, gen_id))
                self.session.current_processing_task = processing_task

            except asyncio.TimeoutError:
                # This block executes every 1 second of silence.
                # Only increment the counter if the AI is NOT speaking.
                if not self.session.is_speaking:
                    idle_seconds_counter += 1

                # If idle time exceeds the threshold, prompt the user.
                if idle_seconds_counter >= SILENCE_THRESHOLD:
                    prompt = await translate_text("Are you still there?", self.session.state.language)
                    await self._stream_tts_response(prompt)
                    # Reset counter to avoid spamming the prompt every second.
                    idle_seconds_counter = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}", exc_info=True)
    
    # ... (the rest of the file remains exactly the same) ...

    async def _process_transcript(self, transcript: str, gen_id: int):
        """Handles the full pipeline from transcript to spoken audio response."""
        self.session.is_speaking = True
        try:
            response_text = ""
            if not self.session.state.patient_id and not self.session.state.patient_info:
                response_text = await self._handle_patient_identification(transcript, gen_id)
            else:
                response_text = await self._handle_conversational_turn(transcript)

            if self.session.generation_id != gen_id: return

            sentences = re.split(r'(?<=[.!?।])\s+', response_text)
            for sentence in sentences:
                if sentence:
                    if self.session.generation_id != gen_id: return
                    tts_task = asyncio.create_task(self._speak_sentence(sentence, gen_id))
                    self.session.current_sentence_tasks.append(tts_task)
                    try:
                        await tts_task
                    except asyncio.CancelledError:
                        break
        
        except Exception as e:
            logger.error(f"Error during transcript processing (gen {gen_id}): {e}", exc_info=True)
        finally:
            if self.session.generation_id == gen_id:
                self.session.is_speaking = False

    async def _handle_patient_identification(self, transcript: str, gen_id: int) -> str:
        """Logic for extracting name, finding patient, and forming a greeting."""
        name_extract_prompt = "You are an expert at extracting a person's full name from a sentence. Respond ONLY with the full name. If no name is present, respond with 'null'."
        extracted_name = await call_llm(f"Sentence: '{transcript}'", name_extract_prompt)
        extracted_name = extracted_name.strip().replace('"', '')
        logger.info(f"ORCHESTRATOR: Extracted name '{extracted_name}' in original language.")
        
        if self.session.generation_id != gen_id: return ""

        if "null" in extracted_name.lower() or len(extracted_name) < 3:
            return await translate_text("I'm sorry, I didn't catch a name. Could you please state your full name clearly?", self.session.state.language)
        
        search_name_in_english = await translate_text(extracted_name, "en-US") if self.session.state.language not in ENGLISH_CODES else extracted_name
        logger.info(f"DATABASE: Searching for patient with English name: '{search_name_in_english}'")
        
        patient_record = await find_closest_patient_by_name(search_name_in_english)
        if self.session.generation_id != gen_id: return ""

        if patient_record:
            correct_name = patient_record.get("name", search_name_in_english)
            history = f"History: {patient_record.get('medical_history', 'N/A')}. Allergy: {patient_record.get('allergy', 'N/A')}"
            translated_history = await translate_text(history, self.session.state.language)
            self.session.load_patient_data(patient_record, translated_history)
            return await translate_text(f"Thank you, {correct_name}. I've found your file. How can I help?", self.session.state.language)
        else:
            self.session.state.patient_info = {"name": search_name_in_english}
            return await translate_text(f"Thank you, {extracted_name}. It looks like this is your first time. How can I help?", self.session.state.language)

    async def _handle_conversational_turn(self, transcript: str) -> str:
        """Logic for a standard conversational exchange with the triage engine."""
        self.session.state.conversation_history.append({"role": "user", "content": transcript})
        ai_response = await triage_engine.get_next_conversational_step(self.session.state, self.session.last_ai_response)
        
        if ai_response.get("is_final", False):
            self.session.state.status = "awaiting_finalization"
            
        return ai_response.get("response_text", "I'm not sure how to respond.")

    async def _speak_sentence(self, text: str, gen_id: int):
        """Streams a single sentence as audio, checking for interruptions."""
        try:
            async for chunk in generate_tts_stream(text):
                if self.session.generation_id != gen_id: break
                if not self.session.websocket_active: break
                await self.websocket.send_bytes(chunk)
        except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
            pass

    async def _stream_tts_response(self, text: str):
        """Utility to stream a simple, non-interruptible text response."""
        try:
            async for chunk in generate_tts_stream(text):
                if not self.session.websocket_active: break
                await self.websocket.send_bytes(chunk)
        except (WebSocketDisconnect, RuntimeError):
            pass

# ==============================================================================
# MAIN APPLICATION AND WEBSOCKET ENDPOINT
# ==============================================================================
@app.get("/")
async def root():
    return {"message": "PHCVue AI Healthcare Assistant is running successfully."}

@app.websocket("/phcvue")
async def conversation_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection and delegates to the ConversationHandler."""
    session_id = str(uuid.uuid4())
    session: SessionManager = None
    try:
        await get_db()
        await websocket.accept()
        logger.info(f"WEBSOCKET: Client connected. Assigning session ID: {session_id}")
        
        config_data = await websocket.receive_json()
        language_code = config_data.get("config", {}).get("language_code")
        if language_code not in SUPPORTED_LANGUAGES:
            await websocket.close(code=4000, reason="Unsupported language.")
            return

        await websocket.send_json({"type": "format", "sampleRate": 24000, "codec": "pcm_s16le"})
        
        session = SessionManager(session_id=session_id, language_code=language_code)
        active_sessions[session_id] = session
        handler = ConversationHandler(websocket, session)
        
        await handler.run()

        if session and session.state.status == "awaiting_finalization":
            await generate_and_save_final_report(session)

    except WebSocketDisconnect:
        logger.info(f"WEBSOCKET: Client for session '{session_id}' disconnected.")
    except Exception as e:
        logger.error(f"WEBSOCKET: An error occurred in session '{session_id}': {e}", exc_info=True)
    finally:
        if session: session.websocket_active = False
        if session_id in active_sessions: del active_sessions[session_id]
        logger.info(f"SESSION END: Cleanup for session '{session_id}'.")

async def generate_and_save_final_report(session: SessionManager):
    """Translates history if needed and saves the final AI suggestion report."""
    logger.info(f"FINALIZING: Preparing report for session '{session.state.session_id}'.")
    
    final_state_for_report = session.state
    if session.state.language not in ENGLISH_CODES:
        translated_history = []
        for entry in session.state.conversation_history:
            translated_content = await translate_text(entry['content'], "en-US")
            translated_history.append({"role": entry['role'], "content": translated_content})
        final_state_for_report = session.state.model_copy(deep=True)
        final_state_for_report.conversation_history = translated_history
        
    doctor_suggestion_report = await triage_engine.generate_doctor_suggestion(final_state_for_report)
    
    final_details = {
        "session_id": session.state.session_id,
        "patient_id": session.state.patient_id,
        "patient_details": session.state.patient_info,
        "conversation_language": session.state.language,
        "full_transcript": session.state.conversation_history,
        "ai_suggestion_for_doctor": doctor_suggestion_report,
        "status": "awaiting_review"
    }
    await save_consultation_details(final_details)
    logger.info(f"FINALIZING: Doctor suggestion saved for session '{session.state.session_id}'.")

@app.get("/health")
async def health_check():
    return {"status": "ok"}