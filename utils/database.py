import os
import re
import datetime
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv
from thefuzz import fuzz
from utils.logger import get_logger

# --- Initialization ---
load_dotenv()
logger = get_logger(__name__)

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "healthcare_db")

# --- Global variables for a single, reusable connection ---
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

async def get_db() -> AsyncIOMotorDatabase:
    """Gets the MongoDB database instance."""
    global _client, _db
    if _db is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI is not set in your .env file.")
        _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client[MONGO_DB_NAME]
        logger.info(f"DATABASE: MongoDB connection established to '{MONGO_DB_NAME}'.")
    return _db

async def close_db():
    """Closes the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("DATABASE: MongoDB connection closed.")

async def find_closest_patient_by_name(spoken_name: str) -> Optional[Dict[str, Any]]:
    """
    Finds the best matching patient and returns the best possible match if any candidates are found.
    """
    db = await get_db()
    
    clean_spoken_name = spoken_name.strip()
    logger.info(f"DATABASE: Performing search for patient resembling '{clean_spoken_name}'.")

    name_parts = re.split(r'\s+', clean_spoken_name)
    regex_pattern = "|".join([re.escape(part) for part in name_parts])
    query = {"name": re.compile(regex_pattern, re.IGNORECASE)}
    
    projection = {"_id": 0, "patient_id": 1, "name": 1, "medical_history": 1, "allergy": 1}

    candidates_cursor = db.patient.find(query, projection)
    candidates = await candidates_cursor.to_list(length=20)

    if not candidates:
        logger.warning(f"DATABASE: No potential candidates found for '{clean_spoken_name}'.")
        return None

    best_match = None
    highest_score = -1
    spoken_name_no_spaces = clean_spoken_name.replace(" ", "").lower()

    for candidate in candidates:
        db_name = candidate.get("name")
        if not db_name: continue

        db_name_no_spaces = db_name.replace(" ", "").lower()
        score = fuzz.ratio(spoken_name_no_spaces, db_name_no_spaces)
        
        if score > highest_score:
            highest_score = score
            best_match = candidate
    
    # --- ✅ ENHANCED LOGGING ---
    if best_match:
        logger.info(f"✅ DATABASE: PATIENT FOUND. Best match for '{clean_spoken_name}' is '{best_match.get('name')}' with score {highest_score}.")
        logger.info("    -> EXTRACTED INFO:")
        logger.info(f"       - Medical History: \"{best_match.get('medical_history', 'N/A')}\"")
        logger.info(f"       - Allergy: \"{best_match.get('allergy', 'N/A')}\"")
        return best_match
    else:
        logger.warning(f"DATABASE: NO MATCH FOUND for '{clean_spoken_name}'. No suitable candidate identified.")
        return None

async def get_patient_history(patient_id: str) -> Optional[dict]:
    """Retrieves the complete document for a patient using their unique patient_id."""
    db = await get_db()
    return await db.patient.find_one({"patient_id": patient_id}, {"_id": 0})

async def save_consultation_details(details: dict):
    """Saves the final consultation details into the 'consultation' collection."""
    db = await get_db()
    session_id = details.get('session_id', 'UNKNOWN_SESSION')
    logger.info(f"DATABASE: Inserting record into 'consultation' for ID '{session_id}'.")
    await db.consultation.insert_one({**details, "timestamp": datetime.datetime.utcnow()})