import requests
import pytest

BASE_URL = 'http://127.0.0.1:5000' 

def register(username, password):
    response = requests.post(f'{BASE_URL}/register', json={'username': username, 'password': password})

    if response.status_code == 201:
        print(response.json().get('message'))

    elif response.status_code ==  400:
        print(response.json().get('error'))

    else:
        print(f'User registration failed. Status code: {response.status_code}')

    response = response.json()
    return response

def login(username, password):
    # Make a POST request to the login endpoint
    response = requests.post(f'{BASE_URL}/login', json={'username': username, 'password': password})
    

    # Check if login was successful
    if response.status_code == 200:
        print(response.json().get('message'))
    else:
        print("Login failed. Please check your credentials.")
    response = response.json()
    return response

@pytest.mark.parametrize("username, password, expected_message", [
    ('khangxlei', 'khang', 'success'),
    ('khangleii', 'password', 'success'),
    ('khangxlei', 'password', 'failure')
])
def test_register(username, password, expected_message):
    upload_response = register(username, password)
    if expected_message == 'success':
        assert upload_response['status_code'] == 201
        assert upload_response['message'] == 'User created successfully'

    else:
        assert upload_response['status_code'] == 400
        assert upload_response['error'] == 'Username already exists'


@pytest.mark.parametrize("username, password, expected_message", [
    ('khangxlei', 'khang', 'success'),
    ('', '', 'empty'),
    ('khang', 'qwertyy', 'failure')
])
def test_login(username, password, expected_message):
    upload_response = login(username, password)
    if expected_message == 'success':
        assert upload_response['status_code'] == 200
        assert upload_response['message'] == 'Login successful'

    elif expected_message == 'empty':
        assert upload_response['status_code'] == 400
        assert upload_response['error'] == 'No data given'

    else:
        assert upload_response['status_code'] == 401
        assert upload_response['message'] == 'Invalid username or password'

if __name__ == "__main__":
    register()
