# Image-Recognition
This is my DIY ML Project 2 as well as Final Project for ENG EC 530. This README provides detailed instructions on how to set up a Python virtual environment for this project and how to run it. Following these steps ensures that all dependencies are managed locally and do not interfere with the global Python environment.

## Demo

Click [here](https://drive.google.com/file/d/1rjupMNurk9r0QkKtmKUJAf1x0EQX9r4f/view?usp=drive_link) to view a demo

## Prerequisites

Before you begin, ensure you have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

## Creating a Virtual Environment

To create a virtual environment within your project directory, follow these steps:

1. **Navigate to your project directory**

   Open your terminal (or command prompt) and navigate to the project directory where you want the virtual environment to be set up.

   ```bash
   cd path/to/your/project-directory
   python -m venv env

2. **Activating the Virtual Environment**
    - On Windows:
       ```bash
       .\env\Scripts\activate
    - On MacOS and Linux:
        ```bash
        source env/bin/activate

    After activation, your command prompt will change to show the name of the activated virtual environment.

3. **Installing Dependencies**

  With the virtual environment activated, you can now install project-specific dependencies using pip:

    pip install -r requirements.txt

  Ensure that requirements.txt is present in your project directory and lists all necessary packages.

4. **Deactivating the Virtual Environment**

   When you are done working within the virtual environment, you can deactivate it by running:

   ```bash
   deactivate

## Getting Started with the Project

1. **Clone the Repository**
   
   First, clone this repository to your local machine:
   ```bash
   git clone https://github.com/Khangxlei/Image-Recognition/

2. **Navigate into the Image-Recognition Directory**
   ```bash
   cd Image-Recognition

3. **Optional: Install SQLite Browser**
  
   Download any software that enables you to view a SQLite Database (I recommend DB Browser). This will help you out later after running the tests to visualize the database.

4. **Run the Flask App**

   Run the Flask application by entering the following command in the terminal:
   ```bash
   python app.py

5. **Run the CLI Tool**

   Run the _run_cli.py file to start the Command Line Interface:

       python run_cli.py
   This will allow you to log in to the project, upload data and model, and name them so you can later on use it. The project will only allow you to access and use the data and models that you have uploaded using your own account.
