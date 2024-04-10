import requests
import json
import pytest
import threading
import queue

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

def process_task(parameters, ids):
    print(f"Processing inference data for user {ids['user_id']} with data {ids['data_id']}")
    data = inference_data(parameters, ids)
    print(f"Accuracy for user {ids['user_id']} with data {ids['data_id']}: {data.get('acc')}")

task_queue = queue.Queue()

def worker():
    while True:
        parameters, user_id, data_id = task_queue.get()
        ids = {'user_id': user_id, 'data_id': data_id}
        upload_response = inference_data(parameters, ids)
        
        assert upload_response['status_code'] == 200

        # Check dataset shapes
        assert upload_response['X.shape'] == [10, 32, 32, 3]
        assert upload_response['y.shape'] == [10,]

        assert upload_response['message'] == 'Model trained and saved successfully.'
        assert upload_response['inference_time_seconds'] < 5

        task_queue.task_done()

# Start worker threads
num_worker_threads = 1  # Define the number of worker threads
for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

parameters = {
    'model_filename': 'model1.h5',
    'loss_name': 'sparse_categorical_crossentropy',
    'img_height': 32,
    'img_width': 32
}

# Fixture to add training requests to the queue
@pytest.fixture(scope='session', autouse=True)
def add_training_requests():
    parameters = {
        'model_filename': 'model1.h5',
        'loss_name': 'sparse_categorical_crossentropy',
        'img_height': 32,
        'img_width': 32
    }

    task_queue.put((parameters, 'khangxlei', 'cifar10'))
    task_queue.put((parameters, 'khang', 'cifar10v2'))
    task_queue.put((parameters, 'khangle', 'cifar10v3'))

    task_queue.join()

# Test function
@pytest.mark.parametrize("parameters, user_id, data_id", [
    (parameters, 'khangxlei', 'cifar10'),
    (parameters, 'khang', 'cifar10v2'),
    (parameters, 'khangle', 'cifar10v3')
])

def test_train(parameters, user_id, data_id):
    pass

# if __name__ == "__main__":


    # Define parameters and IDs for each task
    # parameters = {
    #     'model_filename': 'model1.h5',
    #     'loss_name': 'sparse_categorical_crossentropy',
    #     'img_height': 32,
    #     'img_width': 32
    # }

    # tasks = [
    #     {'user_id': 'khangxlei', 'data_id': 'cifar10'},
    #     {'user_id': 'khang', 'data_id': 'cifar10v2'},
    #     {'user_id': 'khangle', 'data_id': 'cifar10v3'}
    # ]

    # # Enqueue tasks
    # for task_data in tasks:
    #     task_queue.put((parameters, task_data))

    # # Add sentinel values to signal workers to exit after all tasks are processed
    # for _ in range(num_workers):
    #     task_queue.put(None)

    
    # print("All inference data tasks completed")



    # parameters = {
    #     'model_filename': 'model1.h5',
    #     'loss_name': 'sparse_categorical_crossentropy',
    #     'img_height': 32,
    #     'img_width': 32
    # }

    # user_id = 'khangxlei'
    # data_id = 'cifar10'

    # ids = {'user_id': user_id, 'data_id': data_id}

    # data = inference_data(parameters, ids)
    # print(f"accuracy: {data.get('acc')}")

    # user_id = 'khang'
    # data_id = 'cifar10v2'

    # ids = {'user_id': user_id, 'data_id': data_id}

    # data = inference_data(parameters, ids)
    # print(f"accuracy: {data.get('acc')}")

    # user_id = 'khangle'
    # data_id = 'cifar10v3'

    # ids = {'user_id': user_id, 'data_id': data_id}

    # data = inference_data(parameters, ids)
    # print(f"accuracy: {data.get('acc')}")

    

