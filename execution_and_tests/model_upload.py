import requests
import os
import json

# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

def upload_model(folder_path, user_id):
    upload_url = f'{BASE_URL}/upload_models'
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

if __name__ == '__main__':
    # Example usage:
    user_id = 'khangxlei'
    data_folder_path = 'uploads/models'
    upload_response = upload_model(data_folder_path, user_id)
    print('Upload Response:', upload_response)
