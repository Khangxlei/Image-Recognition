import requests
import json
import pytest
import threading
import queue


# Define the URL of your API
BASE_URL = 'http://127.0.0.1:5000'  # Assuming your API is running locally on port 5000

def train_model(parameters, ids):
    # Make a POST request to the /train endpoint
    response = requests.post(f"{BASE_URL}/train", json=parameters, params=ids)

    print(f'response status code: {response.status_code}')
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        message = data['message']  # Extract the message from the response
        print(message)  # or do whatever you want with the message
        return data
         
    else:
        print(f"Error: Unable to train model. Status code: {response.status_code}")
        return response.json()

def process_task(parameters, ids):
    print(f"Processing task for user {ids['user_id']} with data {ids['data_id']}")
    result = train_model(parameters, ids)
    print(result)


parameters = {
    'epochs': 10,
    'model_filename': 'model1.h5',
    'loss_name': 'sparse_categorical_crossentropy',
    'img_height': 32,
    'img_width': 32
}

# Define a queue to hold tasks
task_queue = queue.Queue()

# Worker function to process tasks from the queue
def worker():
    while True:
        parameters, user_id, data_id = task_queue.get()
        ids = {'user_id': user_id, 'data_id': data_id}
        upload_response = train_model(parameters, ids) 

        # Assertions and checks can be added here if needed
        assert upload_response['status_code'] == 200

        # Check dataset shapes
        assert upload_response['X_train.shape'] == [8, 32, 32, 3]
        assert upload_response['y_train.shape'] == [8,]
        assert upload_response['X_test.shape'] == [2, 32, 32, 3]
        assert upload_response['y_test.shape'] == [2,]
        assert upload_response['training_time_secs'] < 5


        task_queue.task_done()

# Start worker threads
num_worker_threads = 3  # Define the number of worker threads
for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

# Fixture to add training requests to the queue
@pytest.fixture(scope='session', autouse=True)
def add_training_requests():
    # user_ids = ['khangxlei', 'khang', 'khangle']
    # data_ids = ['cifar10', 'cifar10v2', 'cifar10v3']

    parameters = {
        'epochs': 10,
        'model_filename': 'model1.h5',
        'loss_name': 'sparse_categorical_crossentropy',
        'img_height': 32,
        'img_width': 32
    }

    task_queue.put((parameters, 'khangxlei', 'cifar10'))
    task_queue.put((parameters, 'khang', 'cifar10v2'))
    task_queue.put((parameters, 'khangle', 'cifar10v3'))
 
    # Wait for the queue to empty before finishing the session
    task_queue.join()

# Test function
@pytest.mark.parametrize("parameters, user_id, data_id", [
    (parameters, 'khangxlei', 'cifar10'),
    (parameters, 'khang', 'cifar10v2'),
    (parameters, 'khangle', 'cifar10v3')
])

def test_train(parameters, user_id, data_id):
    pass  # No need to enqueue tasks here, pytest fixture will take care of it


# if __name__ == "__main__":

    # task_queue = queue.Queue()

    # # Create worker threads
    # num_workers = 3
    # workers = []
    # for _ in range(num_workers):
    #     thread = threading.Thread(target=worker)
    #     thread.start()
    #     workers.append(thread)

    # # Define parameters and IDs for each task
    # parameters = {
    #     'epochs': 10,
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

    # # Add sentinel values to signal workers to exit
    # for _ in range(num_workers):
    #     task_queue.put(None)

  

    # print("All tasks completed")
    
   
    # parameters = {
    # 'epochs': 10,
    # 'model_filename': 'model1.h5',
    # 'loss_name': 'sparse_categorical_crossentropy',
    # 'img_height': 32,
    # 'img_width': 32
    # }

    # user_id = 'khangxlei'
    # data_id = 'cifar10'

    # ids = {'user_id': user_id, 'data_id': data_id}

    # upload_response = train_model(parameters, ids)
    # print(upload_response)

    # user_id = 'khang'
    # data_id = 'cifar10v2'

    # ids = {'user_id': user_id, 'data_id': data_id}
    # upload_response = train_model(parameters, ids)
    # print(upload_response)

    # user_id = 'khangle'
    # data_id = 'cifar10v3'

    # ids = {'user_id': user_id, 'data_id': data_id}
    # upload_response = train_model(parameters, ids)
    # print(upload_response)
