from flask import Blueprint, request, jsonify, current_app
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError


from modules.database import save_train_to_db, get_images_and_labels

train_model = Blueprint('train_model', __name__)

train_model.config = {}

@train_model.record
def record_params(setup_state):
    app = setup_state.app
    train_model.config['MODEL_SAVE_PATH'] = app.config.get('MODEL_SAVE_PATH', 'saved_models')


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 16 * 16, 128)                                 # Adjust the input size based on the pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def preprocess(data, img_height, img_width):
    logging.basicConfig(filename='train.log', level=logging.DEBUG)
    # Step 1: Extract images and labels
    images = np.array([entry['image'] for entry in data])
    labels = [entry['label'] for entry in data]

    # Step 2: Preprocess images
    # Example: Resize images to a fixed size and normalize pixel values
    images = np.array([tf.image.resize(image, (img_height, img_width)) / 255.0 for image in images])
    logging.debug(f'this is images.shape:{images.shape}')

    # Step 3: Convert labels to one-hot encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # integer_encoded = integer_encoded.reshape(-1, 1)  # Reshape for OneHotEncoder
    # onehot_encoder = OneHotEncoder()
    # onehot_encoded = onehot_encoder.fit_transform(integer_encoded).toarray()

    logging.debug(f'this is integer_encoded:{integer_encoded}')
    logging.debug(f'this is integer_encoded.shape:{integer_encoded.shape}')

    # Step 4: Create TensorFlow Dataset objects
    dataset = tf.data.Dataset.from_tensor_slices((images, integer_encoded))

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

    # Split the dataset into training and test sets
    train_dataset = dataset_shuffled.take(train_size)
    test_dataset = dataset_shuffled.skip(train_size)

    logging.debug(f'train_dataset:{train_dataset}')
    logging.debug(f'test_dataset:{test_dataset}')

    X_train, y_train = [], []
    X_test, y_test = [], []
    for batch_images, batch_labels in train_dataset:
        X_train.append(batch_images)
        y_train.append(batch_labels)
    for batch_images, batch_labels in test_dataset:
        X_test.append(batch_images)
        y_test.append(batch_labels)

    X_train.pop(0)
    y_train.pop(0)
    X_test.pop(0)
    y_test.pop(0)

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
    img_height = parameters.get('img_height')
    img_width = parameters.get('img_width')
    epochs = parameters.get('epochs')
    model_filename = parameters.get('model_filename')
    model_path_filename = f'model_uploads/{model_filename}'
    loss_name = parameters.get('loss_name')
    user_id = request.args.get('user_id')
    data_id = request.args.get('data_id')

    # set up default values for hyperparameters
    if epochs == None:
        epochs = 10

    if model_filename == None:
        # make error to say that model was not given
        pass

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
    ##use pytest assert here to ensure size is correct

    # logging.debug(f'X_train: {X_train}')
    # logging.debug(f'X_test: {X_test}')
    # logging.debug(f'y_train: {y_train}')
    # logging.debug(f'y_test: {y_test}')

    # Use logging instead of print statements
    logging.debug(f'This is data_dict: {data_dict}')
    logging.debug(f'This is dataset: {dataset}')

    # Load the model (including architecture, weights, and optimizer state)
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

    model_path = save_model_parameters(loaded_model)

    # Store the .pth file path in the database
    save_train_to_db(model_path, loss, acc, optimizer_type, lr, momentum, epochs, training_time_seconds, training_time_minutes)

    return jsonify({'message': 'Model trained and saved successfully'}), 200





    # model_state_dict = torch.load('model.pth')

    

    # net = Net()

    # criterion_class = getattr(nn, criterion_type)
    # optimizer_class = getattr(optim, optimizer_type)
    
    # criterion = criterion_class()
    # optimizer = optimizer_class(net.parameters(), lr=lr, momentum=momentum)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.to(device)

    # if torch.cuda.is_available():
    #     net.cuda()
    # net.train()

    # loss_per_epoch = []

    # start_time = time.time()

    # for _ in range(epochs):
    #     running_loss = 0.0
    #     for i, data in enumerate(X_train):
    #         ## -- ! code required
    #         PIL_image = X_train[i]['image']
    #         labels = y_train[i]['label']

    #         transform = transforms.ToTensor()
    #         tensor_image = transform(PIL_image).to(device)

    #         logging.debug(f'This is tensor_image.shape: {tensor_image.shape}')

    #         label = torch.tensor([labels])

    #         outputs = net(tensor_image)

    #         """logging.debug(f'This is outputs: {outputs}')
    #         logging.debug(f'This is labels: {label}')"""

    #         loss = criterion(outputs, label)

    #         optimizer.zero_grad()

    #         loss.backward()

    #         optimizer.step()

    #         running_loss += loss.item() 

    #     loss_per_epoch.append(running_loss)

    # end_time = time.time()

    # training_time_seconds = end_time - start_time
    # training_time_minutes = training_time_seconds / 60

    # model_state_dict = net.state_dict()

    # model_path = save_model_parameters(model_state_dict)

    # acc = get_accuracy(net, X_test, y_test)
    # optimizer_type = type(optimizer).__name__
    # criterion_type = type(criterion).__name__

    # # Store the .pth file path in the database
    # save_train_to_db(model_path, acc, optimizer_type, criterion_type, lr, momentum, epochs, training_time_seconds, training_time_minutes)

    # return jsonify({'message': 'Model trained and saved successfully'}), 200

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
    model_filename = f'model_parameters{next_number}.h5'

    # Save model parameters to the .pth file
    model_path = os.path.join(models_folder, model_filename)
    model.save(model_path)
    # torch.save(model_state_dict, model_path)
    
    return model_path
    