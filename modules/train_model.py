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

from modules.database import save_train_to_db, get_images_and_labels

train_model = Blueprint('train_model', __name__)

train_model.config = {}

def preprocess(data, img_height, img_width):
    logging.basicConfig(filename='train.log', level=logging.DEBUG)
    # Step 1: Extract images and labels
    images = np.array([entry['image'] for entry in data])
    labels = [entry['label'] for entry in data]

    # Step 2: Preprocess images
    # Example: Resize images to a fixed size and normalize pixel values
    images = np.array([tf.image.resize(image, (img_height, img_width)) / 255.0 for image in images])
    logging.debug(f'this is images.shape:{images.shape}')
    # Get the size of the dataset
    

    # Step 3: Convert labels to one-hot encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    # Step 4: Create TensorFlow Dataset objects
    dataset = tf.data.Dataset.from_tensor_slices((images, integer_encoded))

    dataset_size = tf.data.experimental.cardinality(dataset)
    logging.debug(f'This is dataset size: {dataset_size}')

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(images))
 
    return dataset

def split_train_test(data, dataset, test_split=0.2):
    logging.basicConfig(filename='train.log', level=logging.DEBUG)

    # Define the size of the training and test sets
    train_split = 1 - test_split
    train_size = int(train_split * len(data))  # 80% for training
    test_size = len(data) - train_size  # Remaining 20% for testing

    # Shuffle the dataset
    dataset_shuffled = dataset.shuffle(buffer_size=len(data))

    logging.debug(f'train_size: {train_size}')
    logging.debug(f'test_size: {test_size}')

    # Split the dataset into training and test sets
    train_dataset = dataset_shuffled.take(train_size)
    test_dataset = dataset_shuffled.skip(train_size)#.take(test_size)

    train_dataset_size = tf.data.experimental.cardinality(train_dataset)
    test_dataset_size = tf.data.experimental.cardinality(test_dataset)

    logging.debug(f'This is train_dataset_size: {train_dataset_size}')
    logging.debug(f'This is test_dataset_size: {test_dataset_size}')

    X_train, y_train = [], []
    X_test, y_test = [], []
    for batch_images, batch_labels in train_dataset:
        X_train.append(batch_images)
        y_train.append(batch_labels)
    for batch_images, batch_labels in test_dataset:
        X_test.append(batch_images)
        y_test.append(batch_labels)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

def get_loss_function(loss_name):
    if loss_name == 'sparse_categorical_crossentropy':
        return SparseCategoricalCrossentropy()
    elif loss_name == 'binary_crossentropy':
        return BinaryCrossentropy()
    elif loss_name == 'mean_squared_error':
        return MeanSquaredError()
    else:
        raise ValueError('Invalid loss function name')


@train_model.route('/train', methods=['POST'])
def train():

    # Configure logging
    logging.basicConfig(filename='train.log', level=logging.DEBUG)
    

    # user inputs
    parameters = request.json 
    epochs = parameters.get('epochs')
    img_height = parameters.get('img_height')
    img_width = parameters.get('img_width')
    
    loss_name = parameters.get('loss_name')
    user_id = request.args.get('user_id')
    data_id = request.args.get('data_id')

    # set up default values for hyperparameters
    if epochs == None:
        epochs = 10
    
    
    # Retrieve image filenames and labels from the database
    data_dict = get_images_and_labels(user_id, data_id)
    

    dataset = preprocess(data_dict, img_height, img_width)

    

    logging.debug(f'Completed preprocessing')

    X_train, y_train, X_test, y_test = split_train_test(data_dict, dataset, test_split=0.2)

    logging.debug(f'Completed train test split')
    logging.debug(f'X_train.shape: {X_train.shape}')
    logging.debug(f'X_test.shape: {X_test.shape}')
    logging.debug(f'y_train.shape: {y_train.shape}')
    logging.debug(f'y_test.shape: {y_test.shape}')

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

    optimizer_config = loaded_model.optimizer.get_config()

    optimizer_type = optimizer_config['name']
    momentum = optimizer_config['ema_momentum']
    lr = optimizer_config['learning_rate']
    
    start_time = time.time()

    loaded_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    end_time = time.time()
    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60

    # Evaluate the loaded model
    results = loaded_model.evaluate(X_test, y_test)
    loss = results[0]
    acc = results[1]

    model_path, trained_model_filename = save_model_parameters(loaded_model)

    # Store the .pth file path in the database
    save_train_to_db(
    model_path,
    loss,
    acc,
    optimizer_type,
    lr,
    momentum,
    epochs,
    training_time_seconds,
    training_time_minutes
    )

    return jsonify({
    'message': (
        f'Model trained and saved successfully.\n'
        f'Trained model file: {trained_model_filename}'
    ),
    'status_code': 200,
    'X_train.shape': X_train.shape,
    'y_train.shape': y_train.shape,
    'X_test.shape': X_test.shape,
    'y_test.shape': y_test.shape,
    'training_time_secs': training_time_seconds
    }), 200


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
    