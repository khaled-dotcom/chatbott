from flask import Blueprint, render_template, request, redirect, url_for, session, current_app
from utils.segmentation_model import segment_image
import os
import tempfile
from datetime import datetime

segmentation_bp = Blueprint('segmentation', __name__)

# YOLO model path
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')

@segmentation_bp.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('segmentation.html', error="No image uploaded")
        file = request.files['image']
        if file.filename == '':
            return render_template('segmentation.html', error="No image selected")
        if file:
            # Generate unique filename for uploaded image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            uploaded_filename = f'uploaded_image_{timestamp}.jpg'
            uploaded_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], uploaded_filename)
            
            # Save uploaded image to a temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                temp_filepath = temp_file.name
            
            # Save a copy of the uploaded image to uploads folder
            file.seek(0)  # Reset file pointer
            file.save(uploaded_filepath)
            
            # Run segmentation
            segmented_image = segment_image(temp_filepath, current_app.config['UPLOAD_FOLDER'], YOLO_MODEL_PATH)
            
            # Clean up temporary file
            os.unlink(temp_filepath)
            
            if segmented_image:
                return render_template('segmentation.html', 
                                    uploaded_image=uploaded_filename, 
                                    segmented_image=segmented_image)
            else:
                return render_template('segmentation.html', error="Error processing image")
    return render_template('segmentation.html')