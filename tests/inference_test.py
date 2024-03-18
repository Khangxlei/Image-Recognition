import requests
import json
import pytest

# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

def inference_data(parameters, ids):
    # Make a POST request to the /inference endpoint
    response = requests.post(f"{BASE_URL}/inference", json=parameters, params=ids)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        message = data.get('message')  # Extract the message from the response
        print(message)  # or do whatever you want with the message
        return data
         
    else:
        print(f"Error: Unable to inference model. Status code: {response.status_code}")
        return response.json()
    
parameters = {
    'model_filename': 'model1.h5',
    'loss_name': 'sparse_categorical_crossentropy',
    'img_height': 32,
    'img_width': 32
}

@pytest.mark.parametrize("parameters, user_id, data_id", [
    (parameters, 'khangxlei', 'cifar10'),
    (parameters, 'khang', 'cifar10v2'),
    (parameters, 'khangle', 'cifar10v3')
])

def test_train(parameters, user_id, data_id):
    ids = {'user_id': user_id, 'data_id': data_id}
    upload_response = inference_data(parameters, ids)
    assert upload_response['status_code'] == 200

    # Check dataset shapes
    assert upload_response['X.shape'] == [10, 32, 32, 3]
    assert upload_response['y.shape'] == [10,]

    assert upload_response['message'] == 'Model trained and saved successfully.'
    assert upload_response['inference_time_seconds'] < 5


if __name__ == "__main__":
    parameters = {
        'model_filename': 'model1.h5',
        'loss_name': 'sparse_categorical_crossentropy',
        'img_height': 32,
        'img_width': 32
    }

    user_id = 'khangxlei'
    data_id = 'cifar10'

    ids = {'user_id': user_id, 'data_id': data_id}

    data = inference_data(parameters, ids)
    print(f"accuracy: {data.get('acc')}")

    user_id = 'khang'
    data_id = 'cifar10v2'

    ids = {'user_id': user_id, 'data_id': data_id}

    data = inference_data(parameters, ids)
    print(f"accuracy: {data.get('acc')}")

    user_id = 'khangle'
    data_id = 'cifar10v3'

    ids = {'user_id': user_id, 'data_id': data_id}

    data = inference_data(parameters, ids)
    print(f"accuracy: {data.get('acc')}")

    

