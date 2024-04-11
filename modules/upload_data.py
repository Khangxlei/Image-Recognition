from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
import logging

from modules.database import save_image_to_db, save_labels_to_db

upload_images = Blueprint('upload_images', __name__)
upload_labels = Blueprint('upload_labels', __name__)

# Set the upload folder and allowed extensions
IMAGE_UPLOAD_FOLDER = 'image_uploads'
LABELS_UPLOAD_FOLDER = 'label_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

upload_images.config = {}
upload_labels.config = {}

logging.basicConfig(filename='upload_images.log', level=logging.DEBUG)
logging.basicConfig(filename='upload_labels.log', level=logging.DEBUG)


@upload_images.record
def record_params(setup_state):
    app = setup_state.app
    upload_images.config['IMAGE_UPLOAD_FOLDER'] = app.config.get('IMAGE_UPLOAD_FOLDER', IMAGE_UPLOAD_FOLDER)
    upload_images.config['ALLOWED_EXTENSIONS'] = app.config.get('ALLOWED_EXTENSIONS', ALLOWED_EXTENSIONS)

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in upload_images.config['ALLOWED_EXTENSIONS']

@upload_images.route('/upload_image', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        print(request.files)
        logging.error('No file part')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    user_id = request.args.get('user_id')
    data_id = request.args.get('data_id')

    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file', 'status_code': 400}), 400
    
    # Ensure the upload directory exists
    upload_folder = upload_images.config['IMAGE_UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Read image file and convert to byte stream
        image_data = file.read()

        # Save data to database
        save_image_to_db(user_id, data_id, filename, image_data)

        file.save(os.path.join(upload_folder, filename))
        return jsonify({'message': 'File uploaded successfully', 'status_code': 200}), 200
    else:
        logging.error('Invalid file format')
        return jsonify({'error': 'Invalid file format', 'status_code': 400}), 400


@upload_labels.record
def record_params(setup_state):
    app = setup_state.app
    upload_labels.config['LABELS_UPLOAD_FOLDER'] = app.config.get('LABELS_UPLOAD_FOLDER', LABELS_UPLOAD_FOLDER)

@upload_labels.route('/upload_labels', methods=['POST'])
def upload_label():
    data = request.json
    logging.debug(f'this is data:{data}')

    user_id = request.args.get('user_id')
    data_id = request.args.get('data_id')
    
    logging.debug(f'this is user_id:{user_id}')
    
    # Parse the JSON string to get image names and labels
    for image_name, label in data.items():
        save_labels_to_db(user_id, data_id, image_name, label)

    folder_path = upload_labels.config['LABELS_UPLOAD_FOLDER']

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Specify the file path
    file_path = os.path.join(folder_path, 'labels.json')
    

    with open(file_path, 'w') as f:
        json.dump(data,f)

    return jsonify({'message': 'Labels uploaded succesfully', 'status_code': 200}), 200