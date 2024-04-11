# Image-Recognition
This is my DIY ML Project 2 for ENG EC 530.

## Creating a virtual environment:
  - Open Terminal/Command Prompt: Open your terminal or command prompt.
  - Navigate to Your Project Directory: Use the cd command to navigate to the directory where you want to create your virtual environment.
  - Create Virtual Environment: Run the following command to create a virtual environment named myenv:
      - python -m venv myenv
  - Replace myenv with whatever name you desire

## Using files:
1. First clone this repository by typing:
    _git clone https://github.com/Khangxlei/Image-Recognition/_
2. Navigate into the Image-Recognition directory
3. Install all the dependencies that are shown in requirements.txt
4. (Optional) Download any software that enables you to view a SQLite Database (I use DB Browser). This will help you out later after running the tests to visualize the database. 
5. Run the Flask app by typing this in the command line:
     - python app.py
6. Split the terminal to have another one running. Navigate into the tests directory where all the test files are.
7. Type this line into the terminal:
     - pytest auth_test.py data_upload_test.py model_upload_test.py train_test.py inference_test.py
     - This ensures that pytest runs these files in order, as we want to test individual modules, and some modules are dependent upon other modules on being run first.
