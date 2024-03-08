import requests
import json

# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

def train_model():
    # Make a POST request to the /train endpoint
    response = requests.post(f"{BASE_URL}/train")
    
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
    train_model()
