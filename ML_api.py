from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json

app = Flask(__name__)

# Set the upload folder and allowed extensions

TRAIN_UPLOAD_FOLDER = 'train_uploads'
LABELS_UPLOAD_FOLDER = 'label_uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app.config['TRAIN_UPLOAD_FOLDER'] = TRAIN_UPLOAD_FOLDER
app.config['LABELS_UPLOAD_FOLDER'] = LABELS_UPLOAD_FOLDER

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_train', methods=['POST'])
def upload_train():
    # Check if the post request has the file part
    if 'file' not in request.files:
        print(request.files)
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    # If the user does not select a file, the browser may send an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    

    # Ensure the upload directory exists
    if not os.path.exists(app.config['TRAIN_UPLOAD_FOLDER']):
        os.makedirs(app.config['TRAIN_UPLOAD_FOLDER'])
        

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join((app.config['TRAIN_UPLOAD_FOLDER']), filename))
        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400

@app.route('/upload_labels', methods=['POST'])
def upload_labels():
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.txt'):
        if not os.path.exists(app.config['LABELS_UPLOAD_FOLDER']):
            os.makedirs(app.config['LABELS_UPLOAD_FOLDER'])
    
        # Save the file to the upload directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['LABELS_UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Parse the text file into a dictionary
        labels_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                image_name, label = line.strip().split(',')
                labels_dict[image_name.strip()] = label.strip()

        print(f'labels_dict: {labels_dict}')

        return jsonify({'labels': labels_dict}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400"""
    
    data = request.json
    
    folder_path = app.config['LABELS_UPLOAD_FOLDER']
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Specify the file path
    file_path = os.path.join(folder_path, 'labels.json')
    
    
    
    with open(file_path, 'w') as f:
        json.dump(data,f)

    return jsonify({'message': 'Labels uploaded succesfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)
