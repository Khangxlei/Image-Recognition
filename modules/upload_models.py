from flask import Blueprint, request, jsonify
import os
import json
from werkzeug.utils import secure_filename


from modules.database import save_model_to_db

upload_models = Blueprint('upload_models', __name__)

MODELS_UPLOAD_FOLDER = 'model_uploads'
ALLOWED_EXTENSIONS = {'pth', 'h5'}

upload_models.config = {}


@upload_models.record
def record_params(setup_state):
    app = setup_state.app
    upload_models.config['MODELS_UPLOAD_FOLDER'] = app.config.get('MODELS_UPLOAD_FOLDER', MODELS_UPLOAD_FOLDER)
    upload_models.config['ALLOWED_EXTENSIONS'] = app.config.get('ALLOWED_EXTENSIONS', ALLOWED_EXTENSIONS)

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in upload_models.config['ALLOWED_EXTENSIONS']

@upload_models.route('/upload_models', methods=['POST'])
def upload_model():    
    if 'file' not in request.files:
        print(request.files)
        return jsonify({'error': 'No file part', 'status_code': 400}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status_code': 400}), 400
    
    # Ensure the upload directory exists
    upload_folder = upload_models.config['MODELS_UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        upload_path = str(os.path.join(upload_folder, filename))
        file.save(upload_path)

        user_id = request.args.get('user_id')
        
        save_model_to_db(upload_path, user_id)
        return jsonify({'message': 'File uploaded successfully', 'status_code': 200}), 200
    else:
        return jsonify({'error': 'Invalid file format', 'status_code': 400}), 400