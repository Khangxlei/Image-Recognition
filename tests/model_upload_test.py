import requests
import os
import pytest

# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

def upload_model(folder_path, user_id):
    upload_url = f'{BASE_URL}/upload_models'
    uploaded_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files = {'file': open(file_path, 'rb')}
            params = {'user_id': user_id}
            response = requests.post(upload_url, files=files, params=params)
            if response.status_code == 200:
                uploaded_files += 1

    response = response.json()
    response['message'] = f'{uploaded_files} files uploaded successfully'
    return response

@pytest.mark.parametrize("folder_path, user_id, expected_uploaded_files", [
    ('../uploads/models', 'khangxlei', 1),  
    ('../uploads/models', 'khang', 1),  
    ('../uploads/models', 'khangle', 1)
])
def test_upload_model(folder_path, user_id, expected_uploaded_files):
    upload_response = upload_model(folder_path, user_id)
    assert upload_response['message'] == f'{expected_uploaded_files} files uploaded successfully'
    assert upload_response['status_code'] == 200


if __name__ == '__main__':
    # Example usage:
    user_id = 'khangxlei'
    data_folder_path = '../uploads/models'
    upload_response = upload_model(data_folder_path, user_id)
    print('Upload Response:', upload_response)

    upload_response = upload_model(data_folder_path, 'khang')
    print('Upload Response:', upload_response)

    upload_response = upload_model(data_folder_path, 'khangle')
    print('Upload Response:', upload_response)
