import os
import aiohttp
from typing import AsyncGenerator
from openai import AsyncOpenAI
from utils.logger import get_logger

logger = get_logger(__name__)

# --- API Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Instantiate the OpenAI client at the module level for efficiency
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Language mapping for clearer translation prompts ---
LANGUAGE_MAP = {
    "en-US": "English",
    "en-IN": "English (India)",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada"
}

# --- Text-to-Speech (TTS) Function (Corrected) ---
async def generate_tts_stream(text: str, chunk_size: int = 2048) -> AsyncGenerator[bytes, None]:
    """
    Generates audio from text using the official OpenAI library and streams it.
    This version uses the recommended 'with_streaming_response' context manager
    to correctly handle the asynchronous audio stream.
    """
    if not OPENAI_API_KEY:
        logger.error("API: OPENAI_API_KEY is not set for TTS.")
        return

    try:
        # ✅ FIX: Use the 'with_streaming_response' context manager for robust streaming.
        async with openai_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            input=text,
            # This is the crucial setting for clear audio matching the frontend.
            response_format="pcm", 
        ) as response:
            # This will now correctly iterate over the asynchronous audio chunks.
            async for chunk in response.iter_bytes(chunk_size=chunk_size):
                yield chunk
            
    except Exception as e:
        logger.error(f"API: OpenAI TTS streaming failed: {e}", exc_info=True)
        # In case of an error, the generator simply stops.
        return

# --- Language Model (LLM) Function (Unchanged) ---
async def call_llm(prompt: str, system_prompt: str, json_mode: bool = False) -> str:
    """
    Calls the OpenAI Chat Completions API with robust error handling using aiohttp.
    """
    if not OPENAI_API_KEY:
        logger.error("API: OPENAI_API_KEY is not set.")
        return "Error: AI service is not configured."

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                # Safely access the response content
                return data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Invalid AI response.")
    except aiohttp.ClientResponseError as e:
        logger.error(f"API: OpenAI LLM call failed with status {e.status}: {e.message}")
        return "Error: The AI model is currently unavailable."
    except Exception as e:
        logger.error(f"API: An unexpected error occurred in call_llm: {e}", exc_info=True)
        return "Error: An unexpected error occurred."

# --- Translation Function (Unchanged) ---
async def translate_text(text: str, target_language_code: str) -> str:
    """
    Translates text to a specified language using a highly specific LLM prompt.
    """
    language_name = LANGUAGE_MAP.get(target_language_code, "English")
    
    system_prompt = (
        f"You are a professional translator. Your sole task is to translate the user's text into {language_name}. "
        "Do not add any extra words, explanations, or apologies. "
        "Your response must be ONLY the translated text."
    )
    
    translated_text = await call_llm(prompt=text, system_prompt=system_prompt)
    
    if "Error:" in translated_text:
        logger.warning(f"Translation failed. Falling back to original text: {text}")
        return text
        
    return translated_text.strip().replace('"', '')