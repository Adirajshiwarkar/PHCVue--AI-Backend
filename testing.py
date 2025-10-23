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
client = AsyncIOMotorClient(MONGO_URI)
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



# ################# previos code
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from motor.motor_asyncio import AsyncIOMotorClient
# from datetime import datetime
# from typing import Optional, Dict, List
# import os
# from dotenv import load_dotenv
# load_dotenv()
# from openai import AsyncOpenAI
# import json
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('chatbot.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # MongoDB connection
# MONGO_URI = "mongodb+srv://mnrusers:pvt98mnr76global54technologies32ltd10@cluster0.rmyuiop.mongodb.net/PHCVue?retryWrites=true&w=majority&appName=Cluster0"
# client = AsyncIOMotorClient(MONGO_URI)
# db = client["PHCVue"]
# consultation_collection = db["consultation"]
# ai_record_collection = db["ai_record"]

# # OpenAI client
# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # In-memory conversation storage (use Redis in production)
# conversation_store: Dict[str, List[Dict]] = {}
# patient_info_store: Dict[str, Dict] = {}

# class ChatRequest(BaseModel):
#     conversation_id: str
#     query: str

# class ChatResponse(BaseModel):
#     response: str
#     conversation_id: str


# async def get_last_10_messages(conversation_id: str) -> List[Dict]:
#     """Get the last 10 messages from the current conversation"""
#     conversation = conversation_store.get(conversation_id, [])
#     # Return last 10 messages
#     return conversation[-10:] if len(conversation) > 10 else conversation


# async def get_patient_consultation_count(patient_id: str) -> int:
#     """Get the number of consultations for a patient"""
#     try:
#         patient_record = await consultation_collection.find_one({"patient_id": patient_id})
#         if not patient_record:
#             logger.info(f"No consultations found for patient_id: {patient_id}")
#             return 0
        
#         # Count consultation fields (consultation_1, consultation_2, etc.)
#         count = 0
#         for key in patient_record.keys():
#             if key.startswith("consultation_"):
#                 count += 1
        
#         logger.info(f"Patient {patient_id} has {count} consultations")
#         return count
#     except Exception as e:
#         logger.error(f"Error getting consultation count for {patient_id}: {e}")
#         return 0


# async def get_latest_consultation_summary(patient_id: str) -> Optional[str]:
#     """Get the AI summary from the most recent consultation"""
#     try:
#         patient_record = await consultation_collection.find_one({"patient_id": patient_id})
#         if not patient_record:
#             logger.info(f"No consultation record found for patient_id: {patient_id}")
#             return None
        
#         # Find the highest consultation number
#         consultation_numbers = []
#         for key in patient_record.keys():
#             if key.startswith("consultation_"):
#                 try:
#                     num = int(key.split("_")[1])
#                     consultation_numbers.append(num)
#                 except:
#                     continue
        
#         if not consultation_numbers:
#             logger.warning(f"No valid consultation numbers found for patient_id: {patient_id}")
#             return None
        
#         latest_num = max(consultation_numbers)
#         latest_consultation = patient_record.get(f"consultation_{latest_num}", {})
#         summary = latest_consultation.get("ai_summary")
        
#         logger.info(f"Retrieved latest consultation summary for patient {patient_id}, consultation {latest_num}")
#         return summary
#     except Exception as e:
#         logger.error(f"Error getting latest consultation summary for {patient_id}: {e}")
#         return None


# async def extract_patient_info_context_aware(conversation_id: str, current_query: str) -> Optional[Dict]:
#     """Use GPT to extract patient information with context from last 10 messages"""
#     try:
#         # Get last 10 messages for context
#         last_10_messages = await get_last_10_messages(conversation_id)
        
#         system_prompt = """You are an assistant that extracts and tracks patient information across a conversation.

# Your task:
# 1. Review the conversation history (last 10 messages) AND the current message
# 2. Extract name, patient_id, and age from ALL messages combined
# 3. Track what information has been provided and what's still missing
# 4. If information is incomplete, inform the user EXACTLY what's missing

# IMPORTANT:
# - Information may be provided across multiple messages (e.g., name in first message, ID in second)
# - Use context from previous messages to build complete information
# - Be specific about what's missing

# Return a JSON object with this structure:
# {
#     "name": "value or null",
#     "patient_id": "value or null", 
#     "age": "value or null",
#     "is_complete": true/false,
#     "missing_fields": ["field1", "field2"],
#     "message_to_user": "friendly message about missing info or null if complete"
# }

# Examples:
# - If user provided name and age but not ID: "Thank you! I have your name and age. I just need your Patient ID to proceed."
# - If user provided ID but not name and age: "I have your Patient ID. Could you please provide your name and age?"
# - If all provided: message_to_user should be null"""

#         # Build context from last 10 messages
#         context_messages = [{"role": "system", "content": system_prompt}]
        
#         if last_10_messages:
#             context_messages.append({
#                 "role": "user", 
#                 "content": f"Previous conversation context:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_10_messages])
#             })
        
#         context_messages.append({
#             "role": "user",
#             "content": f"Current user message: {current_query}"
#         })
        
#         logger.info(f"Extracting patient info for conversation_id: {conversation_id}")
        
#         response = await openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=context_messages,
#             temperature=0
#         )
        
#         result = json.loads(response.choices[0].message.content)
#         logger.info(f"Patient info extraction result: {result}")
        
#         return result
        
#     except Exception as e:
#         logger.error(f"Error extracting patient info for conversation {conversation_id}: {e}")
#         return None


# async def check_conversation_complete(conversation: List[Dict]) -> bool:
#     """Use GPT to determine if enough information has been collected"""
#     system_prompt = """You are a medical consultation assistant. Review the conversation and determine if:
# 1. Enough medical information has been collected (symptoms, duration, severity, etc.)
# 2. The user has indicated they have nothing more to share (said "no" or similar when asked if there's anything else)

# Return ONLY a JSON object: {"complete": true/false, "reason": "brief reason"}"""
    
#     conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    
#     try:
#         logger.info("Checking if conversation is complete")
        
#         response = await openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": conversation_text}
#             ],
#             temperature=0
#         )
        
#         result = json.loads(response.choices[0].message.content)
#         logger.info(f"Conversation complete check: {result}")
#         return result.get("complete", False)
#     except Exception as e:
#         logger.error(f"Error checking conversation completion: {e}")
#         return False


# async def generate_consultation_response(conversation: List[Dict], patient_info: Dict, is_returning: bool = False) -> str:
#     """Generate chatbot response using GPT"""
#     system_prompt = """You are a medical consultation assistant. Your role is to:
# 1. Ask detailed questions about the patient's symptoms and condition
# 2. Gather information like: symptoms, duration, severity, triggers, previous treatments
# 3. Ask follow-up questions based on their responses
# 4. Be empathetic and professional
# 5. NEVER make assumptions or diagnoses
# 6. Once you feel you have enough information, ask: "Is there anything else you'd like to share?"
# 7. Keep responses concise and focused

# Remember: You are collecting information, not providing medical advice."""
    
#     if is_returning:
#         system_prompt += "\nThis is a returning patient. Ask them how they're feeling now compared to their previous visit."
    
#     try:
#         logger.info(f"Generating consultation response. Is returning: {is_returning}")
        
#         messages = [{"role": "system", "content": system_prompt}] + conversation
        
#         response = await openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             temperature=0.7,
#             max_tokens=300
#         )
        
#         response_text = response.choices[0].message.content
#         logger.info(f"Generated response: {response_text[:100]}...")
#         return response_text
#     except Exception as e:
#         logger.error(f"OpenAI API error in generate_consultation_response: {e}")
#         raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


# async def generate_ai_summary(conversation: List[Dict]) -> str:
#     """Generate a short summary of the consultation"""
#     system_prompt = """Create a brief medical summary (2-3 sentences) of this consultation.
# Include: main symptoms, duration, and key concerns.
# Be concise and professional."""
    
#     conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation if msg['role'] == 'user'])
    
#     try:
#         logger.info("Generating AI summary")
        
#         response = await openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": conversation_text}
#             ],
#             temperature=0.5
#         )
        
#         summary = response.choices[0].message.content
#         logger.info(f"AI summary generated: {summary[:100]}...")
#         return summary
#     except Exception as e:
#         logger.error(f"Error generating AI summary: {e}")
#         return "Summary generation failed"


# async def generate_ai_record(conversation: List[Dict], patient_info: Dict) -> str:
#     """Generate detailed markdown report with key points and highlights"""
#     system_prompt = """Create a detailed medical consultation report in markdown format.

# Structure:
# # Patient Consultation Report

# ## Patient Information
# - Name: [name]
# - Patient ID: [id]
# - Age: [age]
# - Date: [date]

# ## Chief Complaints
# [List main symptoms/concerns]

# ## Key Points
# [Bullet points of important information]

# ## Symptom Details
# [Detailed description with duration, severity, triggers]

# ## Medical History Mentioned
# [Any relevant history shared]

# ## Core Assessment
# [Your assessment of what could be the primary concern - this is the ONLY place where you can make clinical observations]

# ## Additional Notes
# [Any other relevant information]

# Be thorough, use proper markdown formatting, and make it easy to read."""
    
#     conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    
#     try:
#         logger.info("Generating AI record/markdown report")
        
#         response = await openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Patient Info: {patient_info}\n\nConversation:\n{conversation_text}"}
#             ],
#             temperature=0.5,
#             max_tokens=1500
#         )
        
#         record = response.choices[0].message.content
#         logger.info("AI record generated successfully")
#         return record
#     except Exception as e:
#         logger.error(f"Error generating AI record: {e}")
#         return "# Report Generation Failed"


# async def save_consultation(patient_id: str, conversation: List[Dict], ai_summary: str, consultation_num: int):
#     """Save consultation to MongoDB"""
#     try:
#         consultation_data = {
#             "conversation": conversation,
#             "ai_summary": ai_summary,
#             "timestamp": datetime.utcnow().isoformat()
#         }
        
#         await consultation_collection.update_one(
#             {"patient_id": patient_id},
#             {"$set": {f"consultation_{consultation_num}": consultation_data}},
#             upsert=True
#         )
        
#         logger.info(f"Consultation saved for patient {patient_id}, consultation number {consultation_num}")
#     except Exception as e:
#         logger.error(f"Error saving consultation for {patient_id}: {e}")
#         raise


# async def save_ai_record(patient_id: str, ai_record_markdown: str, consultation_num: int):
#     """Save AI record to MongoDB"""
#     try:
#         record_data = {
#             "markdown": ai_record_markdown,
#             "timestamp": datetime.utcnow().isoformat()
#         }
        
#         await ai_record_collection.update_one(
#             {"patient_id": patient_id},
#             {"$set": {f"consultation_{consultation_num}": record_data}},
#             upsert=True
#         )
        
#         logger.info(f"AI record saved for patient {patient_id}, consultation number {consultation_num}")
#     except Exception as e:
#         logger.error(f"Error saving AI record for {patient_id}: {e}")
#         raise


# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     conversation_id = request.conversation_id
#     user_query = request.query
    
#     logger.info(f"Received chat request - conversation_id: {conversation_id}, query: {user_query[:100]}...")
    
#     # Initialize conversation if new
#     if conversation_id not in conversation_store:
#         conversation_store[conversation_id] = []
#         patient_info_store[conversation_id] = {}
#         logger.info(f"New conversation initialized: {conversation_id}")
    
#     conversation = conversation_store[conversation_id]
#     patient_info = patient_info_store[conversation_id]
    
#     # Add user message to conversation
#     conversation.append({"role": "user", "content": user_query})
    
#     # Step 1: Check if we have complete patient information
#     if not patient_info.get("patient_id") or not patient_info.get("name") or not patient_info.get("age"):
#         logger.info(f"Patient info incomplete for conversation {conversation_id}. Current info: {patient_info}")
        
#         # First message - ask for patient info
#         if len(conversation) == 1:
#             response_text = "Hi! Welcome to the medical consultation. To proceed, I need the following information:\n- Your name\n- Patient ID\n- Age\n\nPlease provide these details."
#             conversation.append({"role": "assistant", "content": response_text})
#             logger.info("Sent initial greeting message")
#             return ChatResponse(response=response_text, conversation_id=conversation_id)
        
#         # Try to extract patient info using context-aware LLM
#         extraction_result = await extract_patient_info_context_aware(conversation_id, user_query)
        
#         if extraction_result:
#             # Update patient info with whatever was extracted
#             if extraction_result.get("name"):
#                 patient_info["name"] = extraction_result["name"]
#             if extraction_result.get("patient_id"):
#                 patient_info["patient_id"] = extraction_result["patient_id"]
#             if extraction_result.get("age"):
#                 patient_info["age"] = extraction_result["age"]
            
#             patient_info_store[conversation_id] = patient_info
#             logger.info(f"Updated patient info: {patient_info}")
            
#             # Check if complete
#             if extraction_result.get("is_complete"):
#                 logger.info(f"Patient info complete for {patient_info['patient_id']}")
                
#                 # Check if patient has previous consultations
#                 consultation_count = await get_patient_consultation_count(patient_info["patient_id"])
                
#                 if consultation_count > 0:
#                     # Returning patient
#                     previous_summary = await get_latest_consultation_summary(patient_info["patient_id"])
#                     response_text = f"Welcome back, {patient_info['name']}! I see this is your consultation number {consultation_count + 1}.\n\n"
#                     if previous_summary:
#                         response_text += f"Previous visit summary: {previous_summary}\n\n"
#                     response_text += "How are you feeling now? Are you feeling better, or are there new concerns?"
#                     patient_info["consultation_num"] = consultation_count + 1
#                     patient_info["is_returning"] = True
#                     logger.info(f"Returning patient: {patient_info['patient_id']}")
#                 else:
#                     # New patient
#                     response_text = f"Thank you, {patient_info['name']}! This is your first consultation. Let's begin.\n\nWhat brings you here today? Please describe your symptoms or concerns."
#                     patient_info["consultation_num"] = 1
#                     patient_info["is_returning"] = False
#                     logger.info(f"New patient: {patient_info['patient_id']}")
                
#                 conversation.append({"role": "assistant", "content": response_text})
#                 return ChatResponse(response=response_text, conversation_id=conversation_id)
#             else:
#                 # Incomplete - send LLM-generated message about what's missing
#                 response_text = extraction_result.get("message_to_user", 
#                     "I couldn't find all the required information. Please provide:\n- Your name\n- Patient ID\n- Age")
#                 conversation.append({"role": "assistant", "content": response_text})
#                 logger.info(f"Patient info incomplete. Missing: {extraction_result.get('missing_fields', [])}")
#                 return ChatResponse(response=response_text, conversation_id=conversation_id)
#         else:
#             response_text = "I couldn't find all the required information. Please provide:\n- Your name\n- Patient ID\n- Age"
#             conversation.append({"role": "assistant", "content": response_text})
#             logger.warning(f"Failed to extract patient info for conversation {conversation_id}")
#             return ChatResponse(response=response_text, conversation_id=conversation_id)
    
#     # Step 2: Check if conversation is complete
#     is_complete = await check_conversation_complete(conversation)
    
#     if is_complete:
#         logger.info(f"Conversation complete for {patient_info['patient_id']}. Generating summaries...")
        
#         # Generate summaries and save to database
#         ai_summary = await generate_ai_summary(conversation)
#         ai_record_markdown = await generate_ai_record(conversation, patient_info)
        
#         consultation_num = patient_info.get("consultation_num", 1)
        
#         # Save to MongoDB
#         await save_consultation(
#             patient_info["patient_id"],
#             conversation,
#             ai_summary,
#             consultation_num
#         )
        
#         await save_ai_record(
#             patient_info["patient_id"],
#             ai_record_markdown,
#             consultation_num
#         )
        
#         response_text = "Thank you for sharing all this information. Your consultation has been recorded and will be reviewed by our medical team. They will get back to you soon. Take care!"
#         conversation.append({"role": "assistant", "content": response_text})
        
#         logger.info(f"Consultation completed and saved for patient {patient_info['patient_id']}")
        
#         # Clear the conversation store for this session
#         del conversation_store[conversation_id]
#         del patient_info_store[conversation_id]
        
#         return ChatResponse(response=response_text, conversation_id=conversation_id)
    
#     # Step 3: Continue conversation - ask medical questions
#     is_returning = patient_info.get("is_returning", False)
#     response_text = await generate_consultation_response(conversation, patient_info, is_returning)
    
#     conversation.append({"role": "assistant", "content": response_text})
    
#     return ChatResponse(response=response_text, conversation_id=conversation_id)


# @app.get("/health")
# async def health_check():
#     logger.info("Health check requested")
#     return {"status": "healthy"}


# @app.get("/patient/{patient_id}/consultations")
# async def get_patient_consultations(patient_id: str):
#     """Get all consultation records for a patient"""
#     try:
#         logger.info(f"Fetching consultations for patient: {patient_id}")
#         patient_record = await consultation_collection.find_one({"patient_id": patient_id})
        
#         if not patient_record:
#             logger.warning(f"No consultations found for patient: {patient_id}")
#             raise HTTPException(status_code=404, detail="Patient not found")
        
#         patient_record.pop("_id", None)
#         logger.info(f"Retrieved consultations for patient: {patient_id}")
#         return patient_record
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching consultations for {patient_id}: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")


# @app.get("/patient/{patient_id}/ai-records")
# async def get_patient_ai_records(patient_id: str):
#     """Get all AI records (markdown reports) for a patient"""
#     try:
#         logger.info(f"Fetching AI records for patient: {patient_id}")
#         ai_records = await ai_record_collection.find_one({"patient_id": patient_id})
        
#         if not ai_records:
#             logger.warning(f"No AI records found for patient: {patient_id}")
#             raise HTTPException(status_code=404, detail="No AI records found")
        
#         ai_records.pop("_id", None)
#         logger.info(f"Retrieved AI records for patient: {patient_id}")
#         return ai_records
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching AI records for {patient_id}: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")


# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting FastAPI application")
#     uvicorn.run(app, host="0.0.0.0", port=8000)    