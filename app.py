from flask import Flask, session, jsonify
from views.auth import auth_bp
from views.dashboard import dashboard_bp
from views.chatbot import chatbot_bp
from views.image_gen import image_gen_bp
from views.segmentation import segmentation_bp
import os

app = Flask(__name__)
app.secret_key = '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'  # Replace with a secure key

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(chatbot_bp)
app.register_blueprint(image_gen_bp)
app.register_blueprint(segmentation_bp)

# Custom 500 error handler
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error: ' + str(error)}), 500

if __name__ == '__main__':
    app.run(debug=False)
