import asyncio
import os
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from amazon_transcribe.client import TranscribeStreamingClient
from src.handlers.transcription_handler import MyTranscriptEventHandler
from src.session_manager import SessionManager
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

class TranscriptionService:
    """Manages the audio stream from the client to Amazon Transcribe."""

    def __init__(self, session: SessionManager, websocket: WebSocket):
        self.session = session
        self.websocket = websocket
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.transcribe_client = TranscribeStreamingClient(region=aws_region)

    async def start_stream(self):
        session_id = self.session.state.session_id
        logger.info(f"STT_SERVICE: Initializing for session '{session_id}'.")
        
        stream = None  # Initialize stream to None for the finally block
        try:
            logger.info(f"STT_SERVICE: Starting stream to AWS Transcribe for session '{session_id}'.")
            stream = await self.transcribe_client.start_stream_transcription(
                language_code=self.session.state.language,
                media_sample_rate_hz=16000,
                media_encoding="pcm",
            )
            logger.info(f"STT_SERVICE: AWS stream started successfully for session '{session_id}'.")
            
            handler = MyTranscriptEventHandler(stream.output_stream, self.session)
            
            async def audio_sender():
                """Receives audio from the client and forwards it to AWS Transcribe."""
                logger.info(f"STT_SERVICE: Audio sender started for session '{session_id}'.")
                
                try:
                    while self.session.websocket_active:
                        message = await self.websocket.receive()
                        
                        if "bytes" in message:
                            await stream.input_stream.send_audio_event(audio_chunk=message["bytes"])
                        elif message.get("type") == "websocket.disconnect" or message.get("text"):
                            # If client sends a stop message or disconnects, stop the loop.
                            break
                
                except (WebSocketDisconnect, RuntimeError) as e:
                    logger.warning(f"STT_SERVICE: Audio sender stopped due to client disconnect or error: {e}")
                finally:
                    # --- ✅ THIS IS THE FIX ---
                    # When the user disconnects, immediately tell Amazon Transcribe we are done.
                    # This "hangs up the call" and prevents the 15-second timeout.
                    logger.info(f"STT_SERVICE: Closing AWS Transcribe input stream for session '{session_id}'.")
                    if stream and stream.input_stream:
                        await stream.input_stream.end_stream()
                    self.session.websocket_active = False

            await asyncio.gather(handler.handle_events(), audio_sender())

        except Exception as e:
            # This will now catch the timeout only if it happens for another reason
            if "Your request timed out" in str(e):
                 logger.warning(f"STT_SERVICE: AWS timed out for session '{session_id}'. This is expected if the user was silent.")
            else:
                logger.error(f"STT_SERVICE: A critical error occurred during streaming for session '{session_id}': {e}", exc_info=True)
        finally:
            logger.info(f"STT_SERVICE: Stream process fully ended for session '{session_id}'.")