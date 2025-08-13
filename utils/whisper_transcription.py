import whisper
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model (singleton to avoid reloading)
_model = None

def load_whisper_model(model_name="base"):
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            _model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise
    return _model

def transcribe_audio(audio_path, language="en"):
    logger.info(f"Starting transcription for: {audio_path}")
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return "Error: Audio file not found."
    if not os.access(audio_path, os.R_OK):
        logger.error(f"Audio file not readable: {audio_path}")
        return "Error: Audio file not readable."
    try:
        model = load_whisper_model()
        logger.info(f"Transcribing audio: {audio_path}")
        result = model.transcribe(audio_path, language=language)
        logger.info(f"Transcription successful: {result['text']}")
        return result["text"]
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return f"Error: Failed to transcribe audio: {str(e)}"