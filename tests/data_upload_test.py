import requests
import os
import json
import pytest

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
    response = response.json()
    response['message'] = f'{uploaded_files} files uploaded successfully'
    return response

def upload_labels(file_path, user_id, data_id):
    
    upload_url = f'{BASE_URL}/upload_labels'

    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            image_name, label = line.strip().split(',')
            data[image_name.strip()] = label.strip()

    json_data = json.dumps(data)

    headers = {'Content-Type': 'application/json'}

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

# Unit tests
@pytest.mark.parametrize("folder_path, user_id, data_id, expected_uploaded_files", [
    ('../uploads/data/images', 'khangxlei', 'cifar10', 10),  
    ('../uploads/data/images', 'khang', 'cifar10v2', 10),  
    ('../uploads/data/images', 'khangle', 'cifar10v3', 10)
])
def test_upload_image(folder_path, user_id, data_id, expected_uploaded_files):
    upload_response = upload_image(folder_path, user_id, data_id)
    assert upload_response['message'] == f'{expected_uploaded_files} files uploaded successfully'
    assert upload_response['status_code'] == 200


@pytest.mark.parametrize("file_path, user_id, data_id", [
    ('../uploads/data/labels/labels.txt', 'khangxlei', 'cifar10'),
    ('../uploads/data/labels/labels.txt', 'khang', 'cifar10v2'),
    ('../uploads/data/labels/labels.txt', 'khangle', 'cifar10v3')
])

def test_upload_labels(file_path, user_id, data_id):
    upload_response = upload_labels(file_path, user_id, data_id)    
    assert upload_response['message'] == 'Labels uploaded succesfully'
    assert upload_response['status_code'] == 200 


if __name__ == '__main__':
    # Example usage:
    user_id = 'khangxlei'
    data_id = 'cifar10'
    data_folder_path = '../uploads/data/images'
    upload_response = upload_image(data_folder_path, user_id, data_id)
    print('Upload Response:', upload_response)

    upload_response = upload_image(data_folder_path, 'khang', 'cifar10v2')
    print('Upload Response:', upload_response)

    upload_response = upload_image(data_folder_path, 'khangle', 'cifar10v3')
    print('Upload Response:', upload_response)
    

    test_data_path = '../uploads/data/labels/labels.txt'
    upload_response = upload_labels(test_data_path, user_id, data_id)
    print('Upload Response:', upload_response)

    upload_response = upload_labels(test_data_path, 'khang', 'cifar10v2')
    print('Upload Response:', upload_response)

    upload_response = upload_labels(test_data_path, 'khangle', 'cifar10v3')
    print('Upload Response:', upload_response)

    