# # main.py (Updated with corrected idle prompt logic)

# import asyncio
# import os
# import uuid
# import json
# import re
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv

# load_dotenv()

# # --- Import Custom Project Modules ---
# from src.session_manager import SessionManager
# from src.triage_engine import TriageEngine
# from src.services.transcription_service import TranscriptionService
# from utils.database import get_db, close_db, find_closest_patient_by_name, save_consultation_details
# from utils.external_apis import generate_tts_stream, call_llm, translate_text
# from utils.logger import get_logger

# # --- Initialize Logger and Application ---
# logger = get_logger(__name__)
# app = FastAPI(title="PHCVue AI Healthcare Assistant")
# triage_engine = TriageEngine()
# active_sessions: dict[str, SessionManager] = {}

# # --- Add CORS Middleware & Lifecycle Events ---
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
# @app.on_event("shutdown")
# async def shutdown_event():
#     await close_db()

# SUPPORTED_LANGUAGES = ["en-US", "en-IN", "hi-IN", "mr-IN", "ta-IN", "te-IN", "kn-IN"]
# ENGLISH_CODES = ["en-US", "en-IN"]

# # ==============================================================================
# # REFACTORED CONVERSATION HANDLER CLASS
# # ==============================================================================
# class ConversationHandler:
#     """Encapsulates all logic for handling a single WebSocket conversation."""
#     def __init__(self, websocket: WebSocket, session: SessionManager):
#         self.websocket = websocket
#         self.session = session

#     async def run(self):
#         """Main entry point to start the conversation flow."""
#         transcription_service = TranscriptionService(self.session, self.websocket)
#         initial_greeting = await translate_text(
#             "Hello, I am your AI health assistant. Please tell me your full name to begin.", 
#             self.session.state.language
#         )
#         await self._stream_tts_response(initial_greeting)
        
#         await asyncio.gather(
#             self._listen_for_transcripts(), 
#             transcription_service.start_stream()
#         )

#     # --- ✅ THIS IS THE FIX ---
#     async def _listen_for_transcripts(self):
#         """
#         Waits for user transcripts and orchestrates the AI response with robust
#         idle timeout logic.
#         """
#         idle_seconds_counter = 0
#         SILENCE_THRESHOLD = 15  # Fire prompt after 15 seconds of true silence

#         while self.session.websocket_active and self.session.state.status != "awaiting_finalization":
#             try:
#                 # Wait for a transcript, but with a short 1-second timeout.
#                 transcript = await asyncio.wait_for(self.session.transcript_queue.get(), timeout=1.0)

#                 # If we receive a transcript, user is active. Reset counter.
#                 idle_seconds_counter = 0

#                 # Interrupt AI if it's speaking.
#                 if self.session.is_speaking:
#                     await self.session.interrupt()
                
#                 # Process the user's speech in the background.
#                 gen_id = self.session.generation_id
#                 processing_task = asyncio.create_task(self._process_transcript(transcript, gen_id))
#                 self.session.current_processing_task = processing_task

#             except asyncio.TimeoutError:
#                 # This block executes every 1 second of silence.
#                 # Only increment the counter if the AI is NOT speaking.
#                 if not self.session.is_speaking:
#                     idle_seconds_counter += 1

#                 # If idle time exceeds the threshold, prompt the user.
#                 if idle_seconds_counter >= SILENCE_THRESHOLD:
#                     prompt = await translate_text("Are you still there?", self.session.state.language)
#                     await self._stream_tts_response(prompt)
#                     # Reset counter to avoid spamming the prompt every second.
#                     idle_seconds_counter = 0

#             except asyncio.CancelledError:
#                 break
#             except Exception as e:
#                 logger.error(f"Error in orchestrator loop: {e}", exc_info=True)
    
#     # ... (the rest of the file remains exactly the same) ...

#     async def _process_transcript(self, transcript: str, gen_id: int):
#         """Handles the full pipeline from transcript to spoken audio response."""
#         self.session.is_speaking = True
#         try:
#             response_text = ""
#             if not self.session.state.patient_id and not self.session.state.patient_info:
#                 response_text = await self._handle_patient_identification(transcript, gen_id)
#             else:
#                 response_text = await self._handle_conversational_turn(transcript)

#             if self.session.generation_id != gen_id: return

#             sentences = re.split(r'(?<=[.!?।])\s+', response_text)
#             for sentence in sentences:
#                 if sentence:
#                     if self.session.generation_id != gen_id: return
#                     tts_task = asyncio.create_task(self._speak_sentence(sentence, gen_id))
#                     self.session.current_sentence_tasks.append(tts_task)
#                     try:
#                         await tts_task
#                     except asyncio.CancelledError:
#                         break
        
#         except Exception as e:
#             logger.error(f"Error during transcript processing (gen {gen_id}): {e}", exc_info=True)
#         finally:
#             if self.session.generation_id == gen_id:
#                 self.session.is_speaking = False

#     async def _handle_patient_identification(self, transcript: str, gen_id: int) -> str:
#         """Logic for extracting name, finding patient, and forming a greeting."""
#         name_extract_prompt = "You are an expert at extracting a person's full name from a sentence. Respond ONLY with the full name. If no name is present, respond with 'null'."
#         extracted_name = await call_llm(f"Sentence: '{transcript}'", name_extract_prompt)
#         extracted_name = extracted_name.strip().replace('"', '')
#         logger.info(f"ORCHESTRATOR: Extracted name '{extracted_name}' in original language.")
        
#         if self.session.generation_id != gen_id: return ""

#         if "null" in extracted_name.lower() or len(extracted_name) < 3:
#             return await translate_text("I'm sorry, I didn't catch a name. Could you please state your full name clearly?", self.session.state.language)
        
#         search_name_in_english = await translate_text(extracted_name, "en-US") if self.session.state.language not in ENGLISH_CODES else extracted_name
#         logger.info(f"DATABASE: Searching for patient with English name: '{search_name_in_english}'")
        
#         patient_record = await find_closest_patient_by_name(search_name_in_english)
#         if self.session.generation_id != gen_id: return ""

#         if patient_record:
#             correct_name = patient_record.get("name", search_name_in_english)
#             history = f"History: {patient_record.get('medical_history', 'N/A')}. Allergy: {patient_record.get('allergy', 'N/A')}"
#             translated_history = await translate_text(history, self.session.state.language)
#             self.session.load_patient_data(patient_record, translated_history)
#             return await translate_text(f"Thank you, {correct_name}. I've found your file. How can I help?", self.session.state.language)
#         else:
#             self.session.state.patient_info = {"name": search_name_in_english}
#             return await translate_text(f"Thank you, {extracted_name}. It looks like this is your first time. How can I help?", self.session.state.language)

#     async def _handle_conversational_turn(self, transcript: str) -> str:
#         """Logic for a standard conversational exchange with the triage engine."""
#         self.session.state.conversation_history.append({"role": "user", "content": transcript})
#         ai_response = await triage_engine.get_next_conversational_step(self.session.state, self.session.last_ai_response)
        
#         if ai_response.get("is_final", False):
#             self.session.state.status = "awaiting_finalization"
            
#         return ai_response.get("response_text", "I'm not sure how to respond.")

#     async def _speak_sentence(self, text: str, gen_id: int):
#         """Streams a single sentence as audio, checking for interruptions."""
#         try:
#             async for chunk in generate_tts_stream(text):
#                 if self.session.generation_id != gen_id: break
#                 if not self.session.websocket_active: break
#                 await self.websocket.send_bytes(chunk)
#         except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
#             pass

#     async def _stream_tts_response(self, text: str):
#         """Utility to stream a simple, non-interruptible text response."""
#         try:
#             async for chunk in generate_tts_stream(text):
#                 if not self.session.websocket_active: break
#                 await self.websocket.send_bytes(chunk)
#         except (WebSocketDisconnect, RuntimeError):
#             pass

# # ==============================================================================
# # MAIN APPLICATION AND WEBSOCKET ENDPOINT
# # ==============================================================================
# @app.get("/")
# async def root():
#     return {"message": "PHCVue AI Healthcare Assistant is running successfully."}

# @app.websocket("/phcvue")
# async def conversation_endpoint(websocket: WebSocket):
#     """Handles the WebSocket connection and delegates to the ConversationHandler."""
#     session_id = str(uuid.uuid4())
#     session: SessionManager = None
#     try:
#         await get_db()
#         await websocket.accept()
#         logger.info(f"WEBSOCKET: Client connected. Assigning session ID: {session_id}")
        
#         config_data = await websocket.receive_json()
#         language_code = config_data.get("config", {}).get("language_code")
#         if language_code not in SUPPORTED_LANGUAGES:
#             await websocket.close(code=4000, reason="Unsupported language.")
#             return

#         await websocket.send_json({"type": "format", "sampleRate": 24000, "codec": "pcm_s16le"})
        
#         session = SessionManager(session_id=session_id, language_code=language_code)
#         active_sessions[session_id] = session
#         handler = ConversationHandler(websocket, session)
        
#         await handler.run()

#         if session and session.state.status == "awaiting_finalization":
#             await generate_and_save_final_report(session)

#     except WebSocketDisconnect:
#         logger.info(f"WEBSOCKET: Client for session '{session_id}' disconnected.")
#     except Exception as e:
#         logger.error(f"WEBSOCKET: An error occurred in session '{session_id}': {e}", exc_info=True)
#     finally:
#         if session: session.websocket_active = False
#         if session_id in active_sessions: del active_sessions[session_id]
#         logger.info(f"SESSION END: Cleanup for session '{session_id}'.")

# async def generate_and_save_final_report(session: SessionManager):
#     """Translates history if needed and saves the final AI suggestion report."""
#     logger.info(f"FINALIZING: Preparing report for session '{session.state.session_id}'.")
    
#     final_state_for_report = session.state
#     if session.state.language not in ENGLISH_CODES:
#         translated_history = []
#         for entry in session.state.conversation_history:
#             translated_content = await translate_text(entry['content'], "en-US")
#             translated_history.append({"role": entry['role'], "content": translated_content})
#         final_state_for_report = session.state.model_copy(deep=True)
#         final_state_for_report.conversation_history = translated_history
        
#     doctor_suggestion_report = await triage_engine.generate_doctor_suggestion(final_state_for_report)
    
#     final_details = {
#         "session_id": session.state.session_id,
#         "patient_id": session.state.patient_id,
#         "patient_details": session.state.patient_info,
#         "conversation_language": session.state.language,
#         "full_transcript": session.state.conversation_history,
#         "ai_suggestion_for_doctor": doctor_suggestion_report,
#         "status": "awaiting_review"
#     }
#     await save_consultation_details(final_details)
#     logger.info(f"FINALIZING: Doctor suggestion saved for session '{session.state.session_id}'.")

# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# MongoDB connection
# MONGO_URI = ""
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["PHCVue"]
consultation_collection = db["consultation"]
ai_record_collection = db["ai_record"]

# OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation storage (use Redis in production)
conversation_store: Dict[str, List[Dict]] = {}
patient_info_store: Dict[str, Dict] = {}

class ChatRequest(BaseModel):
    conversation_id: str
    query: str

class ChatResponse(BaseModel):
    response: str
    conversation_id: str


async def get_last_10_messages(conversation_id: str) -> List[Dict]:
    """Get the last 10 messages from the current conversation"""
    conversation = conversation_store.get(conversation_id, [])
    # Return last 10 messages
    return conversation[-10:] if len(conversation) > 10 else conversation


async def get_patient_consultation_count(patient_id: str) -> int:
    """Get the number of consultations for a patient"""
    try:
        patient_record = await consultation_collection.find_one({"patient_id": patient_id})
        if not patient_record:
            logger.info(f"No consultations found for patient_id: {patient_id}")
            return 0
        
        # Count consultation fields (consultation_1, consultation_2, etc.)
        count = 0
        for key in patient_record.keys():
            if key.startswith("consultation_"):
                count += 1
        
        logger.info(f"Patient {patient_id} has {count} consultations")
        return count
    except Exception as e:
        logger.error(f"Error getting consultation count for {patient_id}: {e}")
        return 0


async def get_latest_consultation_summary(patient_id: str) -> Optional[str]:
    """Get the AI summary from the most recent consultation"""
    try:
        patient_record = await consultation_collection.find_one({"patient_id": patient_id})
        if not patient_record:
            logger.info(f"No consultation record found for patient_id: {patient_id}")
            return None
        
        # Find the highest consultation number
        consultation_numbers = []
        for key in patient_record.keys():
            if key.startswith("consultation_"):
                try:
                    num = int(key.split("_")[1])
                    consultation_numbers.append(num)
                except:
                    continue
        
        if not consultation_numbers:
            logger.warning(f"No valid consultation numbers found for patient_id: {patient_id}")
            return None
        
        latest_num = max(consultation_numbers)
        latest_consultation = patient_record.get(f"consultation_{latest_num}", {})
        summary = latest_consultation.get("ai_summary")
        
        logger.info(f"Retrieved latest consultation summary for patient {patient_id}, consultation {latest_num}")
        return summary
    except Exception as e:
        logger.error(f"Error getting latest consultation summary for {patient_id}: {e}")
        return None


async def extract_patient_info_context_aware(conversation_id: str, current_query: str) -> Optional[Dict]:
    """Use GPT to extract patient information with context from last 10 messages"""
    try:
        # Get last 10 messages for context
        last_10_messages = await get_last_10_messages(conversation_id)
        
        system_prompt = """You are an assistant that extracts and tracks patient information across a conversation.

Your task:
1. Review the conversation history (last 10 messages) AND the current message
2. Extract name, patient_id, and age from ALL messages combined
3. Track what information has been provided and what's still missing
4. If information is incomplete, inform the user EXACTLY what's missing

IMPORTANT:
- Information may be provided across multiple messages (e.g., name in first message, ID in second)
- Use context from previous messages to build complete information
- Be specific about what's missing

Return a JSON object with this structure:
{
    "name": "value or null",
    "patient_id": "value or null", 
    "age": "value or null",
    "is_complete": true/false,
    "missing_fields": ["field1", "field2"],
    "message_to_user": "friendly message about missing info or null if complete"
}

Examples:
- If user provided name and age but not ID: "Thank you! I have your name and age. I just need your Patient ID to proceed."
- If user provided ID but not name and age: "I have your Patient ID. Could you please provide your name and age?"
- If all provided: message_to_user should be null"""

        # Build context from last 10 messages
        context_messages = [{"role": "system", "content": system_prompt}]
        
        if last_10_messages:
            context_messages.append({
                "role": "user", 
                "content": f"Previous conversation context:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_10_messages])
            })
        
        context_messages.append({
            "role": "user",
            "content": f"Current user message: {current_query}"
        })
        
        logger.info(f"Extracting patient info for conversation_id: {conversation_id}")
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context_messages,
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"Patient info extraction result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting patient info for conversation {conversation_id}: {e}")
        return None


async def check_conversation_complete(conversation: List[Dict]) -> bool:
    """Use GPT to determine if enough information has been collected"""
    system_prompt = """You are a medical consultation assistant. Review the conversation and determine if the consultation should be completed.

A consultation is COMPLETE if ANY of these conditions are met:

1. **User explicitly wants to end**: User says "no", "that's it", "nothing else", "done", "bye", etc. when asked if there's anything more
2. **Sufficient information collected**: You have gathered:
   - Main symptoms/complaints
   - Basic details (when it started, how severe, any patterns)
   - At least 2-3 exchanges of medical information
3. **User is unresponsive or giving very short answers**: If user gives minimal responses like "no", "yes", "ok" repeatedly without elaborating

IMPORTANT: Don't make the consultation too long. If you have basic symptom information and the user seems done (short answers, says "no", "that's it", etc.), mark it as complete.

Return ONLY a JSON object: {"complete": true/false, "reason": "brief reason"}

Examples of COMPLETE:
- User: "No" (after being asked if there's anything else)
- User: "That's it" 
- User: "Nothing more"
- After collecting basic symptoms and user gives minimal follow-up

Examples of INCOMPLETE:
- User is actively describing symptoms
- User is answering detailed questions
- First or second message about symptoms"""
    
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    
    try:
        logger.info("Checking if conversation is complete")
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"Conversation complete check: {result}")
        return result.get("complete", False)
    except Exception as e:
        logger.error(f"Error checking conversation completion: {e}")
        return False


async def generate_consultation_response(conversation: List[Dict], patient_info: Dict, is_returning: bool = False) -> str:
    """Generate chatbot response using GPT"""
    system_prompt = """You are a medical consultation assistant. Your role is to:
1. Ask detailed questions about the patient's symptoms and condition
2. Gather information like: symptoms, duration, severity, triggers, previous treatments
3. Ask follow-up questions based on their responses
4. Be empathetic and professional
5. NEVER make assumptions or diagnoses
6. After gathering basic information (2-3 exchanges), ask: "Is there anything else you'd like to share about your condition?"
7. Keep responses concise and focused
8. If user gives very short answers or seems done, acknowledge and prepare to wrap up

IMPORTANT: Don't make consultations too long. After collecting basic symptom information:
- Main complaint
- When it started
- Severity level
- Any key patterns or triggers

Ask if they have anything else to add, then be ready to conclude.

Remember: You are collecting information, not providing medical advice."""
    
    if is_returning:
        system_prompt += "\n\nThis is a returning patient. Ask them how they're feeling now compared to their previous visit, but keep it brief."
    
    try:
        logger.info(f"Generating consultation response. Is returning: {is_returning}")
        
        messages = [{"role": "system", "content": system_prompt}] + conversation
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"Generated response: {response_text[:100]}...")
        return response_text
    except Exception as e:
        logger.error(f"OpenAI API error in generate_consultation_response: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


async def generate_ai_summary(conversation: List[Dict]) -> str:
    """Generate a short summary of the consultation"""
    system_prompt = """Create a brief medical summary (2-3 sentences) of this consultation.
Include: main symptoms, duration, and key concerns.
Be concise and professional."""
    
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation if msg['role'] == 'user'])
    
    try:
        logger.info("Generating AI summary")
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.5
        )
        
        summary = response.choices[0].message.content
        logger.info(f"AI summary generated: {summary[:100]}...")
        return summary
    except Exception as e:
        logger.error(f"Error generating AI summary: {e}")
        return "Summary generation failed"


async def generate_ai_record(conversation: List[Dict], patient_info: Dict) -> str:
    """Generate detailed markdown report with key points and highlights"""
    system_prompt = """Create a detailed medical consultation report in markdown format.

Structure:
# Patient Consultation Report

## Patient Information
- Name: [name]
- Patient ID: [id]
- Age: [age]
- Date: [date]

## Chief Complaints
[List main symptoms/concerns]

## Key Points
[Bullet points of important information]

## Symptom Details
[Detailed description with duration, severity, triggers]

## Medical History Mentioned
[Any relevant history shared]

## Core Assessment
[Your assessment of what could be the primary concern - this is the ONLY place where you can make clinical observations]

## Additional Notes
[Any other relevant information]

Be thorough, use proper markdown formatting, and make it easy to read."""
    
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    
    try:
        logger.info("Generating AI record/markdown report")
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Patient Info: {patient_info}\n\nConversation:\n{conversation_text}"}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        
        record = response.choices[0].message.content
        logger.info("AI record generated successfully")
        return record
    except Exception as e:
        logger.error(f"Error generating AI record: {e}")
        return "# Report Generation Failed"


async def save_consultation(patient_id: str, conversation: List[Dict], ai_summary: str, consultation_num: int):
    """Save consultation to MongoDB"""
    try:
        consultation_data = {
            "conversation": conversation,
            "ai_summary": ai_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await consultation_collection.update_one(
            {"patient_id": patient_id},
            {"$set": {f"consultation_{consultation_num}": consultation_data}},
            upsert=True
        )
        
        logger.info(f"Consultation saved for patient {patient_id}, consultation number {consultation_num}")
    except Exception as e:
        logger.error(f"Error saving consultation for {patient_id}: {e}")
        raise


async def save_ai_record(patient_id: str, ai_record_markdown: str, consultation_num: int):
    """Save AI record to MongoDB"""
    try:
        record_data = {
            "markdown": ai_record_markdown,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await ai_record_collection.update_one(
            {"patient_id": patient_id},
            {"$set": {f"consultation_{consultation_num}": record_data}},
            upsert=True
        )
        
        logger.info(f"AI record saved for patient {patient_id}, consultation number {consultation_num}")
    except Exception as e:
        logger.error(f"Error saving AI record for {patient_id}: {e}")
        raise


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    conversation_id = request.conversation_id
    user_query = request.query
    
    logger.info(f"Received chat request - conversation_id: {conversation_id}, query: {user_query[:100]}...")
    
    # Initialize conversation if new
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = []
        patient_info_store[conversation_id] = {}
        logger.info(f"New conversation initialized: {conversation_id}")
    
    conversation = conversation_store[conversation_id]
    patient_info = patient_info_store[conversation_id]
    
    # Add user message to conversation
    conversation.append({"role": "user", "content": user_query})
    
    # Step 1: Check if we have complete patient information
    if not patient_info.get("patient_id") or not patient_info.get("name") or not patient_info.get("age"):
        logger.info(f"Patient info incomplete for conversation {conversation_id}. Current info: {patient_info}")
        
        # First message - ask for patient info
        if len(conversation) == 1:
            response_text = "Hi! Welcome to the medical consultation. To proceed, I need the following information:\n- Your name\n- Patient ID\n- Age\n\nPlease provide these details."
            conversation.append({"role": "assistant", "content": response_text})
            logger.info("Sent initial greeting message")
            return ChatResponse(response=response_text, conversation_id=conversation_id)
        
        # Try to extract patient info using context-aware LLM
        extraction_result = await extract_patient_info_context_aware(conversation_id, user_query)
        
        if extraction_result:
            # Update patient info with whatever was extracted
            if extraction_result.get("name"):
                patient_info["name"] = extraction_result["name"]
            if extraction_result.get("patient_id"):
                patient_info["patient_id"] = extraction_result["patient_id"]
            if extraction_result.get("age"):
                patient_info["age"] = extraction_result["age"]
            
            patient_info_store[conversation_id] = patient_info
            logger.info(f"Updated patient info: {patient_info}")
            
            # Check if complete
            if extraction_result.get("is_complete"):
                logger.info(f"Patient info complete for {patient_info['patient_id']}")
                
                # Check if patient has previous consultations
                consultation_count = await get_patient_consultation_count(patient_info["patient_id"])
                
                if consultation_count > 0:
                    # Returning patient
                    previous_summary = await get_latest_consultation_summary(patient_info["patient_id"])
                    response_text = f"Welcome back, {patient_info['name']}! I see this is your consultation number {consultation_count + 1}.\n\n"
                    if previous_summary:
                        response_text += f"Previous visit summary: {previous_summary}\n\n"
                    response_text += "How are you feeling now? Are you feeling better, or are there new concerns?"
                    patient_info["consultation_num"] = consultation_count + 1
                    patient_info["is_returning"] = True
                    logger.info(f"Returning patient: {patient_info['patient_id']}")
                else:
                    # New patient
                    response_text = f"Thank you, {patient_info['name']}! This is your first consultation. Let's begin.\n\nWhat brings you here today? Please describe your symptoms or concerns."
                    patient_info["consultation_num"] = 1
                    patient_info["is_returning"] = False
                    logger.info(f"New patient: {patient_info['patient_id']}")
                
                conversation.append({"role": "assistant", "content": response_text})
                return ChatResponse(response=response_text, conversation_id=conversation_id)
            else:
                # Incomplete - send LLM-generated message about what's missing
                response_text = extraction_result.get("message_to_user", 
                    "I couldn't find all the required information. Please provide:\n- Your name\n- Patient ID\n- Age")
                conversation.append({"role": "assistant", "content": response_text})
                logger.info(f"Patient info incomplete. Missing: {extraction_result.get('missing_fields', [])}")
                return ChatResponse(response=response_text, conversation_id=conversation_id)
        else:
            response_text = "I couldn't find all the required information. Please provide:\n- Your name\n- Patient ID\n- Age"
            conversation.append({"role": "assistant", "content": response_text})
            logger.warning(f"Failed to extract patient info for conversation {conversation_id}")
            return ChatResponse(response=response_text, conversation_id=conversation_id)
    
    # Step 2: Check if conversation is complete
    # Also check for forced completion if conversation is too long or user seems done
    conversation_length = len([msg for msg in conversation if msg['role'] == 'user'])
    
    # Check last user messages for completion signals
    last_user_messages = [msg['content'].lower().strip() for msg in conversation[-3:] if msg['role'] == 'user']
    completion_signals = ['no', 'nothing', 'thats it', "that's it", 'done', 'bye', 'nope', 'nothing else']
    user_wants_to_end = any(msg in completion_signals or len(msg.split()) <= 2 for msg in last_user_messages[-1:])
    
    is_complete = await check_conversation_complete(conversation)
    
    # Force completion if conversation has enough exchanges and user seems done
    if conversation_length >= 4 and user_wants_to_end:
        is_complete = True
        logger.info(f"Forcing completion: conversation_length={conversation_length}, user_wants_to_end={user_wants_to_end}")
    
    if is_complete:
        logger.info(f"Conversation complete for {patient_info['patient_id']}. Generating summaries...")
        
        # Generate summaries and save to database
        ai_summary = await generate_ai_summary(conversation)
        ai_record_markdown = await generate_ai_record(conversation, patient_info)
        
        consultation_num = patient_info.get("consultation_num", 1)
        
        # Save to MongoDB
        await save_consultation(
            patient_info["patient_id"],
            conversation,
            ai_summary,
            consultation_num
        )
        
        await save_ai_record(
            patient_info["patient_id"],
            ai_record_markdown,
            consultation_num
        )
        
        response_text = "Thank you for sharing all this information. Your consultation has been recorded and will be reviewed by our medical team. They will get back to you soon. Take care!"
        conversation.append({"role": "assistant", "content": response_text})
        
        logger.info(f"Consultation completed and saved for patient {patient_info['patient_id']}")
        
        # Clear the conversation store for this session
        del conversation_store[conversation_id]
        del patient_info_store[conversation_id]
        
        return ChatResponse(response=response_text, conversation_id=conversation_id)
    
    # Step 3: Continue conversation - ask medical questions
    is_returning = patient_info.get("is_returning", False)
    response_text = await generate_consultation_response(conversation, patient_info, is_returning)
    
    conversation.append({"role": "assistant", "content": response_text})
    
    return ChatResponse(response=response_text, conversation_id=conversation_id)


@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}


@app.get("/patient/{patient_id}/consultations")
async def get_patient_consultations(patient_id: str):
    """Get all consultation records for a patient"""
    try:
        logger.info(f"Fetching consultations for patient: {patient_id}")
        patient_record = await consultation_collection.find_one({"patient_id": patient_id})
        
        if not patient_record:
            logger.warning(f"No consultations found for patient: {patient_id}")
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient_record.pop("_id", None)
        logger.info(f"Retrieved consultations for patient: {patient_id}")
        return patient_record
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching consultations for {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/patient/{patient_id}/ai-records")
async def get_patient_ai_records(patient_id: str):
    """Get all AI records (markdown reports) for a patient"""
    try:
        logger.info(f"Fetching AI records for patient: {patient_id}")
        ai_records = await ai_record_collection.find_one({"patient_id": patient_id})
        
        if not ai_records:
            logger.warning(f"No AI records found for patient: {patient_id}")
            raise HTTPException(status_code=404, detail="No AI records found")
        
        ai_records.pop("_id", None)
        logger.info(f"Retrieved AI records for patient: {patient_id}")
        return ai_records
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching AI records for {patient_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)


