import asyncio
from typing import Optional
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from src.session_manager import SessionManager
from utils.logger import get_logger

logger = get_logger(__name__)
SILENCE_THRESHOLD_MS = 600

class MyTranscriptEventHandler(TranscriptResultStreamHandler):
    """Handles transcription events with advanced barge-in and silence detection."""
    def __init__(self, output_stream, session: SessionManager):
        super().__init__(output_stream)
        self.session = session
        self.silence_timer_task: Optional[asyncio.Task] = None

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        if not self.session.websocket_active: return
            
        results = transcript_event.transcript.results
        if not results or not results[0].alternatives: return

        transcript = results[0].alternatives[0].transcript
        if not transcript.strip(): return

        session_id = self.session.state.session_id

        if results[0].is_partial:
            if self.session.is_speaking and len(transcript.split()) >= 2:
                logger.info(f"[{session_id}] 🎤 Partial transcript triggered barge-in: '{transcript}...'.")
                await self.session.interrupt()
            return

        logger.debug(f"[{session_id}] 💬 Final segment received: '{transcript}'. Buffering...")
        self.session.transcript_buffer.append(transcript)

        if self.silence_timer_task and not self.silence_timer_task.done():
            self.silence_timer_task.cancel()

        self.silence_timer_task = asyncio.create_task(self._finalize_after_silence())

    async def _finalize_after_silence(self):
        """Waits for a brief silence, then combines buffered transcripts."""
        await asyncio.sleep(SILENCE_THRESHOLD_MS / 1000.0)
        
        if self.session.transcript_buffer:
            complete_utterance = " ".join(self.session.transcript_buffer).strip()
            self.session.transcript_buffer.clear()
            
            logger.info(f"[{self.session.state.session_id}] ✅ Final Utterance after silence: '{complete_utterance}'.")
            await self.session.transcript_queue.put(complete_utterance)