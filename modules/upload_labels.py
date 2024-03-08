from flask import Blueprint, request, jsonify
import os
import json

from modules.database import save_labels_to_db

upload_labels = Blueprint('upload_labels', __name__)

LABELS_UPLOAD_FOLDER = 'label_uploads'

upload_labels.config = {}


@upload_labels.record
def record_params(setup_state):
    app = setup_state.app
    upload_labels.config['LABELS_UPLOAD_FOLDER'] = app.config.get('LABELS_UPLOAD_FOLDER', LABELS_UPLOAD_FOLDER)

@upload_labels.route('/upload_labels', methods=['POST'])
def upload_label():
    data = request.json
    
    # Parse the JSON string to get image names and labels
    for image_name, label in data.items():
        save_labels_to_db(image_name, label)

    folder_path = upload_labels.config['LABELS_UPLOAD_FOLDER']

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Specify the file path
    file_path = os.path.join(folder_path, 'labels.json')
    

    with open(file_path, 'w') as f:
        json.dump(data,f)

    return jsonify({'message': 'Labels uploaded succesfully'}), 200
