import requests
import os
import json

# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

# Function to upload an image
def upload_image(folder_path, user_id):
    upload_url = f'{BASE_URL}/upload_image'
    uploaded_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files = {'file': open(file_path, 'rb')}
            data = {'user_id': user_id}
            response = requests.post(upload_url, files=files, data=data)
            if response.status_code == 200:
                uploaded_files += 1
    return {'message': f'{uploaded_files} files uploaded successfully'}

def upload_lables(file_path):
    
    upload_url = f'{BASE_URL}/upload_labels'

    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            image_name, label = line.strip().split(',')
            data[image_name.strip()] = label.strip()

    

    
    json_data = json.dumps(data)

    headers = {'Content-Type': 'application/json'}

    # Make a POST request to the API endpoint with the JSON data
    response = requests.post(upload_url, data=json_data, headers=headers)


    # Check the response
    if response.status_code == 200:
        print("Data saved successfully!")
    else:
        print("Error:", response.text)

    #response = requests.post(upload_url, files=file_contents)
    return response.json()
    
    
# Function to detect faces in an image
def detect_faces(file_path):
    detect_faces_url = f'{BASE_URL}/detect_faces'
    files = {'file': open(file_path, 'rb')}
    response = requests.post(detect_faces_url, files=files)
    return response.json()

if __name__ == '__main__':
    # Example usage:
    user_id = 'khangxlei'
    data_folder_path = 'data/images'
    upload_response = upload_image(data_folder_path, user_id)
    print('Upload Response:', upload_response)

    test_data_path = 'data/labels/labels.txt'
    upload_response = upload_lables(test_data_path)
    print('Upload Response:', upload_response)