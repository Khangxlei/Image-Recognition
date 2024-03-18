from flask import Blueprint, request, jsonify
import torch.nn as nn
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import os
import time
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError


from modules.database import get_images_and_labels
from modules.train_model import get_loss_function, preprocess

inference_data = Blueprint('inference_data', __name__)

inference_data.config = {}

def Dataloader(data, dataset):
    logging.basicConfig(filename='train.log', level=logging.DEBUG)

    # Shuffle the dataset
    dataset_shuffled = dataset.shuffle(buffer_size=len(data))

    X, y = [], []

    for batch_images, batch_labels in dataset_shuffled:
        X.append(batch_images)
        y.append(batch_labels)

    X = np.array(X)
    y = np.array(y)

    return X, y

@inference_data.route('/inference', methods=['POST'])
def inference():
    # Configure logging
    logging.basicConfig(filename='inference.log', level=logging.DEBUG)

    # user inputs
    parameters = request.json 
    img_height = parameters.get('img_height')
    img_width = parameters.get('img_width')
    loss_name = parameters.get('loss_name')

    user_id = request.args.get('user_id')
    data_id = request.args.get('data_id')


    # Retrieve image filenames and labels from the database
    data_dict = get_images_and_labels(user_id, data_id)

    dataset = preprocess(data_dict, img_height, img_width)

    logging.debug(f'Completed preprocessing')

    X, y = Dataloader(data_dict, dataset)

    logging.debug(f'Completed loading data')

    logging.debug(f'This is X.shape: {X.shape}')
    logging.debug(f'This is y.shape: {y.shape}')

    # Use logging instead of print statements
    logging.debug(f'This is data_dict: {data_dict}')
    logging.debug(f'This is dataset: {dataset}')

    # Load the model (including architecture, weights, and optimizer state)
    model_filename = parameters.get('model_filename')
    model_path_filename = f'model_uploads/{model_filename}'
    if model_filename == None:
        # make error to say that model was not given
        pass

    loaded_model = tf.keras.models.load_model(model_path_filename)

    loss_function = get_loss_function(loss_name)

    loaded_model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    
    start_time = time.time()

    results = loaded_model.evaluate(X, y)

    end_time = time.time()

    

    inference_time_seconds = end_time - start_time
    inference_time_minutes = inference_time_seconds / 60

    # Evaluate the loaded model
    loss = results[0]
    acc = results[1]

    predictions = loaded_model.predict(X)

    logging.debug(f'Completed inferencing')

    predictions = predictions.tolist()

    results = {
    'inference_time_seconds': inference_time_seconds, 
    'inference_time_minutes': inference_time_minutes, 
    'loss': loss, 'acc': acc, 
    'predictions': predictions, 
    'message': f'Model trained and saved successfully.',
    'status_code': 200,
    'X.shape': X.shape,
    'y.shape': y.shape
    }
    return results, 200


def get_accuracy(net, X_test, y_test):
    logging.basicConfig(filename='train.log', level=logging.DEBUG)
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(X_test):
            PIL_image = X_test[i]['image']
            labels = y_test[i]['label']

            transform = transforms.ToTensor()
            tensor_image = transform(PIL_image).to(device)
            label = torch.tensor([labels])

            
            outputs = net(tensor_image)

            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    acc = 100 * correct / total

    logging.debug(f'Accuracy of the net_cifar on the test images:: {acc}%')
    return acc


def save_model_parameters(model):
    # Define the folder where you'll save the .pth files
    models_folder = 'trained_models'
    os.makedirs(models_folder, exist_ok=True)

    # Get the list of existing .pth files in the folder
    existing_files = [file for file in os.listdir(models_folder) if file.endswith('.h5')]

    # Determine the next available number for the filename
    next_number = len(existing_files) + 1 

    # Construct the filename with the next available number
    trained_model_filename = f'model_parameters{next_number}.h5'

    # Save model parameters to the .pth file
    model_path = os.path.join(models_folder, trained_model_filename)
    model.save(model_path)
    # torch.save(model_state_dict, model_path)
    
    return model_path, trained_model_filename
    