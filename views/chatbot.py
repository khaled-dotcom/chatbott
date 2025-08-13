from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, current_app
import requests
import os
import logging
import time

chatbot_bp = Blueprint('chatbot', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODAL_API_URL = "https://alnemrabdullah2--qwen2-5-medical-web-response.modal.run/"

@chatbot_bp.route('/chatbot')
def chatbot():
    if not session.get('logged_in'):
        logger.info("User not logged in, redirecting to login")
        return redirect(url_for('auth.login'))
    logger.info("Rendering chatbot page")
    return render_template('chatbot.html')

@chatbot_bp.route('/chatbot/ask', methods=['POST'])
def ask():
    if not session.get('logged_in'):
        logger.info("Unauthorized access to /chatbot/ask")
        return jsonify({'error': 'Unauthorized'}), 401
    user_input = request.json.get('instruction')
    if not user_input.strip():
        logger.error("Empty question provided")
        return jsonify({'error': 'Please provide a valid question'}), 400
    try:
        logger.info(f"Sending question to Qwen API: {user_input}")
        response = requests.post(MODAL_API_URL, json={"instruction": user_input})
        response.raise_for_status()
        logger.info("Qwen API response received")
        return jsonify({'response': response.json().get('response', 'No response received')})
    except requests.RequestException as e:
        logger.error(f"Qwen API error: {str(e)}", exc_info=True)
        return jsonify({'error': f"Error contacting the model: {str(e)}"}), 500

@chatbot_bp.route('/chatbot/transcribe', methods=['POST'])
def transcribe():
    logger.info("Received request for /chatbot/transcribe")
    try:
        if not session.get('logged_in'):
            logger.info("Unauthorized access to /chatbot/transcribe")
            return jsonify({'error': 'Unauthorized'}), 401
        
        if 'audio' not in request.files:
            logger.error("No audio file provided in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({'error': 'No audio file selected'}), 400
        
        allowed_extensions = {'wav', 'mp3', 'm4a'}
        file_ext = audio_file.filename.rsplit('.', 1)[-1].lower() if '.' in audio_file.filename else ''
        if file_ext not in allowed_extensions:
            logger.error(f"Unsupported audio format: {audio_file.filename}")
            return jsonify({'error': 'Unsupported audio format. Use WAV, MP3, or M4A.'}), 400
        
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
        logger.info(f"Using upload folder: {upload_folder}")
        os.makedirs(upload_folder, exist_ok=True)
        
        temp_filename = f"audio_{session.get('username', 'user')}_{int(time.time())}.wav"
        temp_filepath = os.path.join(upload_folder, temp_filename)
        
        logger.info(f"Saving audio to {temp_filepath}")
        audio_file.save(temp_filepath)
        
        if not os.path.exists(temp_filepath):
            logger.error(f"Failed to save audio file: {temp_filepath}")
            return jsonify({'error': 'Failed to save audio file'}), 500
        
        try:
            from utils.whisper_transcription import transcribe_audio
            logger.info("Calling transcribe_audio")
            transcription = transcribe_audio(temp_filepath)
            logger.info(f"Transcription result: {transcription}")
        except ImportError as e:
            logger.error(f"Failed to import transcribe_audio: {str(e)}", exc_info=True)
            return jsonify({'error': f"Transcription module error: {str(e)}"}), 500
        
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            logger.info(f"Deleted temporary file: {temp_filepath}")
        
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            logger.info(f"Deleted temporary file after error: {temp_filepath}")
        return jsonify({'error': f"Transcription failed: {str(e)}"}), 500