from flask import Blueprint, jsonify, current_app
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

from modules.database import save_model_to_db, get_images_and_labels

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

@train_model.route('/train', methods=['POST'])
def train():
    # Configure logging
    logging.basicConfig(filename='train.log', level=logging.DEBUG)

    # Retrieve image filenames and labels from the database
    data_dict = get_images_and_labels()

    # Use logging instead of print statements
    logging.debug(f'This is data_dict: {data_dict}')

    # Extracting X and y from data_dict
    X = [{'image_id': d['image_id'], 'image': d['image']} for d in data_dict]
    y = [{'image_id': d['image_id'], 'label': d['label']} for d in data_dict]

    # Splitting X and y into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    logging.debug(f'This is X_train: {X_train}')
    logging.debug(f'This is X_test: {X_test}')
    logging.debug(f'This is y_train: {y_train}')
    logging.debug(f'This is y_test: {y_test}')

    ##use pytest assert here to ensure size is correct

    net = Net()
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    if torch.cuda.is_available():
        net.cuda()
    net.train()

    for _ in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(X_train):
            ## -- ! code required
            PIL_image = X_train[i]['image']
            labels = y_train[i]['label']

            transform = transforms.ToTensor()
            tensor_image = transform(PIL_image).to(device)

            logging.debug(f'This is tensor_image.shape: {tensor_image.shape}')

            label = torch.tensor([labels])

            outputs = net(tensor_image)

            """logging.debug(f'This is outputs: {outputs}')
            logging.debug(f'This is labels: {label}')"""

            loss = criterion(outputs, label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item() 

    model_state_dict = net.state_dict()

    model_path = save_model_parameters(model_state_dict)

    # Store the .pth file path in the database
    save_model_to_db(model_path)

    return jsonify({'message': 'Model trained and saved successfully'}), 200


def save_model_parameters(model_state_dict):
    # Define the folder where you'll save the .pth files
    models_folder = 'saved_models'
    os.makedirs(models_folder, exist_ok=True)

    # Get the list of existing .pth files in the folder
    existing_files = [file for file in os.listdir(models_folder) if file.endswith('.pth')]

    # Determine the next available number for the filename
    next_number = len(existing_files) + 1

    # Construct the filename with the next available number
    model_filename = f'model_parameters{next_number}.pth'

    # Save model parameters to the .pth file
    model_path = os.path.join(models_folder, model_filename)
    torch.save(model_state_dict, model_path)
    
    return model_path
    