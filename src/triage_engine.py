import json
from src.models import ConsultationState
from utils.external_apis import call_llm, translate_text
from utils.logger import get_logger

logger = get_logger(__name__)

class TriageEngine:
    """
    This class is the central AI brain for the project, handling logic for
    both the patient-facing conversation (LLM 1) and the final doctor
    analysis (LLM 2).
    """

    def _is_serious_condition(self, text: str) -> bool:
        """Detects keywords that indicate a potentially serious medical condition."""
        serious_keywords = [
            "chest pain", "crushing pain", "pain in arm", "can't breathe",
            "cannot breathe", "difficulty breathing", "slurred speech", 
            "sudden weakness", "severe headache", "heavy bleeding", "suicidal"
        ]
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in serious_keywords):
            logger.critical(f"TRIAGE: Serious condition keyword detected: '{text_lower}'.")
            return True
        return False

    # --- CORRECTED: Added 'last_ai_response' to the function definition ---
    async def get_next_conversational_step(self, state: ConsultationState, last_ai_response: str) -> dict:
        """
        Handles the voice-to-voice part of the interaction. Its only job is to ask
        clarifying questions and decide when the conversation is complete.
        """
        logger.info("TRIAGE (LLM 1): Getting next conversational step...")
        last_user_query = state.conversation_history[-1].get("content", "") if state.conversation_history else ""

        if self._is_serious_condition(last_user_query):
            emergency_text = "Based on your symptoms, this could be a serious issue. Please consult a doctor immediately."
            translated_text = await translate_text(emergency_text, state.language)
            return {"response_text": translated_text, "is_final": True, "is_serious": True}

        system_prompt = (
            "You are a compassionate AI Nurse. Your ONLY goal is to ask clarifying questions about the patient's symptoms. "
            "If you are interrupted, the user's new query takes priority unless they say 'carry on'. "
            "After a few questions, summarize what you've heard and ask 'Is there anything else?'. "
            "If they say no, set 'is_final' to true. "
            "Your response MUST be a valid JSON object with 'response_text' and 'is_final' (boolean) keys."
        )
        
        prompt = (
            f"Patient's Translated History (for context):\n{state.translated_history}\n\n"
            f"Last AI Response (what you were saying before interruption):\n'{last_ai_response}'\n\n"
            f"Current Conversation:\n{json.dumps(state.conversation_history)}\n\n"
            "Based on the last user message, what is the next step?"
        )

        response_str = await call_llm(prompt, system_prompt, json_mode=True)
        
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            logger.error("TRIAGE (LLM 1): Failed to parse JSON from conversational LLM.")
            return {"response_text": "I'm having a little trouble understanding. Could you please repeat that?", "is_final": False}

    # --- LLM 2: The Doctor's Analyst ---
    async def generate_doctor_suggestion(self, state: ConsultationState) -> dict:
        """
        Works in the backend AFTER the patient conversation ends. Its sole purpose
        is to generate a structured, private suggestion for the human doctor.
        """
        logger.info("TRIAGE (LLM 2): Generating doctor-facing suggestion...")
        
        system_prompt = (
            "You are a clinical decision support AI. Your audience is a qualified human doctor. "
            "Analyze the complete patient conversation and their history. "
            "Your response MUST be a single, valid JSON object with three keys: "
            "'patient_summary', 'potential_diagnosis' (a list), and "
            "'ai_suggestion'. "
            "The 'ai_suggestion' text MUST begin with the disclaimer: 'This is an AI-generated suggestion. Please verify clinically.'"
        )
        
        prompt = (
            f"Patient Information:\n{json.dumps(state.patient_info)}\n\n"
            f"Translated Past History:\n{state.translated_history}\n\n"
            f"Full Conversation Transcript:\n{json.dumps(state.conversation_history)}\n\n"
            "Generate your structured clinical report for the doctor based on all the above information."
        )

        response_str = await call_llm(prompt, system_prompt, json_mode=True)
        
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            logger.error("TRIAGE (LLM 2): Failed to parse JSON for doctor report.")
            return {
                "patient_summary": "Error during AI analysis.",
                "potential_diagnosis": [],
                "ai_suggestion": "This is an AI-generated suggestion. Please verify clinically. The AI failed to produce a structured output; manual review of the transcript is required."
            }