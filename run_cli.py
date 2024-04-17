import sys
import os

# Add the tests directory to the path to be able to import auth_test
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

# Import the main function from auth_test
from tests.auth_test import *
from tests.data_upload_test import *
from tests.model_upload_test import *
from tests.train_test import *
from tests.inference_test import *

def login_menu():
    print("\nWelcome to the System")
    print("1. Register")
    print("2. Login")
    print("3. Exit")
    return int(input("Choose an option (1-3): "))

def post_login_menu():
    print("\nMain Menu")
    print("1. Upload Data")
    print("2. Upload Model")
    print("3. Train Model")
    print("4. Run Inference")
    print("5. Logout")
    return int(input("Select an option (1-4): "))

def get_credentials():
    username = input("Username: ")
    password = input("Password: ")
    return username, password

def get_integer_input(prompt):
    """ Helper function to get a validated integer input from the user. """
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_string_input(prompt):
    """ Helper function to get a string input from the user. """
    return input(prompt)

def worker(start_event):
    start_event.wait()  # Wait for the signal to start processing
    while True:
        parameters, user_id, data_id = task_queue.get()
        ids = {'user_id': user_id, 'data_id': data_id}
        upload_response = train_model(parameters, ids)
        print(upload_response['message'])
        task_queue.task_done()

def worker_inference(start_event_inference):
    start_event_inference.wait()
    while True:
        parameters, user_id, data_id = task_queue_inference.get()
        ids = {'user_id': user_id, 'data_id': data_id}
        upload_response = inference_data(parameters, ids)
        print(upload_response['message'])
        task_queue_inference.task_done()

# Function to get training request details from the user
def get_training_request():
    epochs = int(input("Enter number of epochs: "))
    model_filename = input("Enter model filename: ")
    loss_name = input("Enter loss function name: ")
    img_height = int(input("Enter image height: "))
    img_width = int(input("Enter image width: "))
    
    parameters = {
        'epochs': epochs,
        'model_filename': model_filename,
        'loss_name': loss_name,
        'img_height': img_height,
        'img_width': img_width
    }
    
    data_id = input("Enter data ID: ")
    
    return parameters, data_id


def get_inference_request():
    model_filename = input("Enter model filename: ")
    loss_name = input("Enter loss function name: ")
    img_height = int(input("Enter image height: "))
    img_width = int(input("Enter image width: "))
    
    parameters = {
        'model_filename': model_filename,
        'loss_name': loss_name,
        'img_height': img_height,
        'img_width': img_width
    }
    
    data_id = input("Enter data ID: ")
    
    return parameters, data_id



def main():
    user_logged_in = False
    username = None
    password = None

    global task_queue
    task_queue = queue.Queue()
    start_event = threading.Event()  # Event to control the start of the workers

    global task_queue_inference
    task_queue_inference = queue.Queue()
    start_event_inference = threading.Event() 

    num_worker_threads = 1  # Define the number of worker threads
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker, args=(start_event,))
        t.daemon = True
        t.start()

    num_worker_threads = 1  
    for i in range(num_worker_threads):
        ti = threading.Thread(target=worker_inference, args=(start_event_inference,))
        ti.daemon = True
        ti.start()
    
    while True:
        if not user_logged_in:
            choice = login_menu()
            if choice == 1:
                username, password = get_credentials()
                register(username, password)
            elif choice == 2:
                username, password = get_credentials()
                response = login(username, password)  # Assuming login returns True on success
                if response['status_code'] == 200:
                    user_logged_in = True
            elif choice == 3:
                print("Exiting the system.")
                break
        else:
            choice = post_login_menu()
            if choice == 1:
                folder_path = input("Please enter the file path of your data with no / at the beginning or end: ")
                data_id = input("Please give a label for your data: ")

                image_folder_path = folder_path + '/images'
                image_upload_response = upload_image(image_folder_path, username, data_id)
                print(image_upload_response['message'])

                label_folder_path = folder_path + '/labels/labels.txt'
                label_upload_response = upload_labels(label_folder_path, username, data_id)
                print(label_upload_response['message'])

            elif choice == 2:
                folder_path = input("Please enter the file path of your model with no / at the beginning or end: ")
                model_upload_response = upload_model(folder_path, username)

                print(model_upload_response['message'])
            elif choice == 3:
                # Loop to collect user inputs and add them to the queue
                while True:
                    print("\nEnter the details for a new training request, or type 'start' to begin processing.")
                    user_input = input("Type 'new' to add a new request or 'start' to begin processing: ")
                    if user_input.lower() == 'new':
                        parameters, data_id = get_training_request()
                        task_queue.put((parameters, username, data_id))
                    elif user_input.lower() == 'start':
                        start_event.set()  # Signal the workers to start processing
                        break
                    else:
                        print("Invalid input, please enter 'new' or 'start'.")
                
                # Wait for all tasks in the queue to be completed
                task_queue.join()
                print("All training requests have been processed.")

            elif choice == 4:
                # Loop to collect user inputs and add them to the queue
                while True:
                    print("\nEnter the details for a new inference request, or type 'start' to begin processing.")
                    user_input = input("Type 'new' to add a new request or 'start' to begin processing: ")
                    if user_input.lower() == 'new':
                        parameters, data_id = get_inference_request()
                        task_queue_inference.put((parameters, username, data_id))
                    elif user_input.lower() == 'start':
                        start_event_inference.set()  # Signal the workers to start processing
                        break
                    else:
                        print("Invalid input, please enter 'new' or 'start'.")
                
                # Wait for all tasks in the queue to be completed
                task_queue_inference.join()
                print("All inference requests have been processed.")

            elif choice == 5:
                print("Logging out...")
                user_logged_in = False


if __name__ == "__main__":
    main()
