from flask import Flask, request, jsonify, g
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
import sqlite3
import logging





app = Flask(__name__)

# Set the upload folder and allowed extensions

IMAGE_UPLOAD_FOLDER = 'image_uploads'
LABELS_UPLOAD_FOLDER = 'label_uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

DATABASE_FILE = 'database.db'



app.config['IMAGE_UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER
app.config['LABELS_UPLOAD_FOLDER'] = LABELS_UPLOAD_FOLDER
app.config['DATABASE'] = 'database.db'

# Initialize SQLite database
DATABASE = 'database.db'

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()



# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        print(request.files)
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not os.path.exists(app.config['IMAGE_UPLOAD_FOLDER']):
        os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'])

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
        file.save(file_path)

        save_image_to_db(filename)
        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400
    
def save_image_to_db(filename):
    db = get_db()
    db.execute('INSERT INTO images (filename) VALUES (?)', [filename])
    db.commit()

    

def save_labels_to_db(image_name, label):
    db = get_db()
    db.execute('INSERT INTO labels (image_name, label) VALUES (?, ?)', (image_name, label))
    db.commit()


@app.route('/upload_labels', methods=['POST'])
def upload_labels():
    data = request.json
    
    # Parse the JSON string to get image names and labels
    for image_name, label in data.items():
        save_labels_to_db(image_name, label)

    folder_path = app.config['LABELS_UPLOAD_FOLDER']
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Specify the file path
    file_path = os.path.join(folder_path, 'labels.json')
    
    # save_image_to_db(data)

    with open(file_path, 'w') as f:
        json.dump(data,f)

    return jsonify({'message': 'Labels uploaded succesfully'}), 200

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 16 * 16, 128)                                 # Adjust the input size based on the pooling

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 6)


    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.view(-1, 32 * 16 * 16)

        x = self.fc1(x)

        x = self.fc2(x)

        x = self.fc3(x)

        return x

net = Net()

def get_images_and_labels():
    db = get_db()
    cursor = db.cursor()

    sql_query = """
    SELECT images.filename, labels.label
    FROM images
    JOIN labels ON images.filename = labels.image_name
    """

    # Execute the SQL query
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    
    images_and_labels = [{'filename': row[0], 'label': row[1]} for row in rows]
    
    return images_and_labels


@app.route('/train', methods=['POST'])
def train_model():

    # Retrieve image filenames and labels from the database
    images_and_labels = get_images_and_labels()
    

    # images_and_labels = [{'filename': '1.jpg', 'label': 'Lionel Messi'}, {'filename': '2.jpg', 'label': 'Cristiano Ronaldo'}, \
    # {'filename': '3.jpg', 'label': 'LeBron James'}, {'filename': '4.jpg', 'label': 'Michael Jordan'}]

    # Use logging instead of print statements
    logging.debug(f'This is images_and_labels: {images_and_labels}')

    
    # print(images_and_labels)

    # # Get uploaded image data and labels
    # image_data = []
    # image_data_path = 'data/images/'
    # for filename in os.listdir(image_data_path):
    #     image_data.append(os.path.join(image_data_path, filename))


    # labels = request.files.get('labels')

    # # Preprocess image data and labels if necessary

    # # Split data into training and testing sets (80-20 split)
    # X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

    





    # # # Train your model
    # # trained_model = .train_model(X_train, y_train)

    # # # Evaluate your model
    # # evaluation_results = your_model_module.evaluate_model(trained_model, X_test, y_test)

    # # # Optionally, save the trained model
    # # your_model_module.save_model(trained_model, 'trained_model.pkl')

    # return jsonify({'message': 'Model trained successfully', 'evaluation_results': evaluation_results})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
