import requests
import os
import json

# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

# Function to upload an image
def upload_image(folder_path, user_id, data_id):
    upload_url = f'{BASE_URL}/upload_image'
    uploaded_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files = {'file': open(file_path, 'rb')}
            params = {'user_id': user_id, 'data_id':data_id}
            response = requests.post(upload_url, files=files, params=params)
            if response.status_code == 200:
                uploaded_files += 1
    return {'message': f'{uploaded_files} files uploaded successfully'}

def upload_lables(file_path, user_id, data_id):
    
    upload_url = f'{BASE_URL}/upload_labels'

    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            image_name, label = line.strip().split(',')
            data[image_name.strip()] = label.strip()

    json_data = json.dumps(data)

    headers = {'Content-Type': 'application/json'}

    # json_data['user_id'] = user_id
    # json_data['data_id'] =  data_id

    params = {'user_id': user_id, 'data_id': data_id}

    # Make a POST request to the API endpoint with the JSON data
    response = requests.post(upload_url, data=json_data, headers=headers, params=params)


    # Check the response
    if response.status_code == 200:
        print("Data saved successfully!")
    else:
        print("Error:", response.text)

    #response = requests.post(upload_url, files=file_contents)
    return response.json()
    

if __name__ == '__main__':
    # Example usage:
    user_id = 'khangxlei'
    data_id = 'cifar10'
    data_folder_path = 'uploads/data/images'
    upload_response = upload_image(data_folder_path, user_id, data_id)
    print('Upload Response:', upload_response)

    test_data_path = 'uploads/data/labels/labels.txt'
    upload_response = upload_lables(test_data_path, user_id, data_id)
    print('Upload Response:', upload_response)