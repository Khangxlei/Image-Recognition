import requests
import json

# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

def train_model(parameters, ids):
    # Make a POST request to the /train endpoint
    response = requests.post(f"{BASE_URL}/train", json=parameters, params=ids)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract data from the response
        data = response.json()
        
        # Process the data (e.g., train your model using the data)
        # for item in data:
        #     print(f"Image: {item['filename']}, Label: {item['label']}")
        
        print("Training completed successfully.")
    else:
        print(f"Error: Unable to train model. Status code: {response.status_code}")

if __name__ == "__main__":
    parameters = {
    'epochs': 10,
    'model_filename': 'model1.h5',
    'loss_name': 'sparse_categorical_crossentropy',
    'img_height': 32,
    'img_width': 32
    }

    user_id = 'khangxlei'
    data_id = 'cifar10'

    ids = {'user_id': user_id, 'data_id': data_id}

    train_model(parameters, ids)
