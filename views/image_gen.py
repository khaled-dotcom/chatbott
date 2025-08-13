from flask import Blueprint, render_template, request
import requests
from io import BytesIO
from PIL import Image
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

image_gen_bp = Blueprint('image_gen', __name__)

# Endpoint for the diffusion model
# put your API URL here
ENDPOINT_URL = ""

# Directory to save generated images
UPLOAD_FOLDER = os.path.join('static', 'Uploads')

@image_gen_bp.route('/image_gen', methods=['GET', 'POST'])
def generate_image():
    error = None
    symptoms = None
    image_path = None

    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        logging.debug(f"Received symptoms: {symptoms}")
        if not symptoms:
            error = "Please enter symptoms."
            logging.error("No symptoms provided")
        else:
            payload = {"prompt": symptoms}
            try:
                logging.debug(f"Sending request to {ENDPOINT_URL} with payload: {payload}")
                response = requests.post(ENDPOINT_URL, json=payload, timeout=120)
                logging.debug(f"API response status: {response.status_code}, content-type: {response.headers.get('content-type')}, content-length: {len(response.content)}")
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        error = f"API returned invalid content type: {content_type}"
                        logging.error(error)
                        if content_type.startswith('application/json'):
                            logging.debug(f"API response JSON: {response.text}")
                    elif len(response.content) < 100:
                        error = "API returned empty or invalid image data"
                        logging.error(error)
                    else:
                        try:
                            image_data = BytesIO(response.content)
                            image = Image.open(image_data)
                            image.verify()
                            image = Image.open(image_data)
                            filename = f"generated_image_{int(time.time())}.png"
                            image_path = os.path.join(UPLOAD_FOLDER, filename)
                            logging.debug(f"Attempting to save image to {image_path}")
                            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                            image.save(image_path, format='PNG')
                            # Normalize path for URL
                            image_path = os.path.join('Uploads', filename).replace('\\', '/')
                            logging.debug(f"Image saved successfully, path: {image_path}")
                        except Exception as e:
                            error = f"Failed to process or save image: {str(e)}"
                            logging.error(error)
                else:
                    error = f"API error: {response.status_code} - {response.text}"
                    logging.error(error)
            except requests.exceptions.ReadTimeout:
                error = "The image generation server took too long to respond. Please try again later."
                logging.error(error)
            except requests.exceptions.ConnectionError:
                error = "Unable to connect to the image generation server. Please check your network or try again later."
                logging.error(error)
            except Exception as e:
                error = f"An unexpected error occurred: {str(e)}"
                logging.error(error)

    return render_template('image_gen.html', error=error, symptoms=symptoms, image_path=image_path)