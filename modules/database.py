import sqlite3
from flask import g, current_app
from PIL import Image
import io



def init_db(app):
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(current_app.config['DATABASE'])
    return db

def close_connection(exception=None):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def get_images_and_labels():
    db = get_db()
    cursor = db.cursor()

    # Define the SQL query
    sql_query = """
        SELECT labels.image_id, labels.label, images.image_data
        FROM images
        JOIN labels ON images.id = labels.image_id
    """

    # Execute the SQL query
    cursor.execute(sql_query)

    # Fetch all the results
    results = cursor.fetchall()

    # Define a list to hold the data
    image_data_list = []

    # Process the results
    for row in results:
        # Extract data from the row
        image_name, label, image_data = row
        
        # Convert BLOB data to image
        image = Image.open(io.BytesIO(image_data))
        
        # Create a dictionary to hold the image data
        image_data_dict = {
            'image_id': image_name,
            'label': label,
            'image': image  # Store the image object directly
        }
        
        # Append the dictionary to the list
        image_data_list.append(image_data_dict)

    # Return the structured data
    return image_data_list

def get_image_id(image_name):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT id FROM images WHERE filename = ?', (image_name,))
    result = cursor.fetchone()
    if result:
        return result[0]  # Return the image_id
    else:
        print("Image with filename '{}' not found.".format(image_name))
        return None
    
def save_image_to_db(filename, image_data):
    db = get_db()
    db.execute('INSERT INTO images (filename, image_data) VALUES (?, ?)', (filename, image_data))
    db.commit()

def save_labels_to_db(image_name, label):
    image_id = get_image_id(image_name)
    if image_id is not None:
        db = get_db()
        # Insert into labels table with image_id and label
        db.execute('INSERT INTO labels (image_id, label) VALUES (?, ?)', (image_id, label))
        db.commit()

def save_model_to_db(model_path):
    db = get_db()
    db.execute('INSERT INTO models (model_file_path) VALUES (?)', [model_path])
    db.commit()


