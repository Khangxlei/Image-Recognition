from functools import wraps
import sqlite3
from flask import Blueprint, request, session, redirect, url_for, request, g, jsonify, current_app
import logging
import json
import bcrypt
from cryptography.fernet import Fernet

from modules.database import get_db


auth = Blueprint('auth', __name__)

auth.config = {}

logging.basicConfig(filename='auth.log', level=logging.DEBUG)


key = Fernet.generate_key()
cipher_suite = Fernet(key)

#Functions for data sanitization
def hash_password(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def encrypt_data(data):
    if isinstance(data, str):
        data = data.encode()  # Convert string to bytes if necessary
    return cipher_suite.encrypt(data)

def decrypt_data(encrypted_data):
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        return decrypted_data.decode()  # Decode bytes to string
    except:
        # Handle invalid token error
        print("Invalid token: Unable to decrypt the data.")
        return None

# Function to authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect(current_app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT username, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user:
        stored_encrypted_hashed_password = user[1]  # Assuming password is stored at index 1
        # decrypt password
        stored_hashed_password = decrypt_data(stored_encrypted_hashed_password)
        if isinstance(password, str):
            password = password.encode('utf-8')  # Convert string password to bytes
        if isinstance(stored_hashed_password, str):
            stored_hashed_password = stored_hashed_password.encode('utf-8')  # Convert stored hashed password to bytes
        try:
            if bcrypt.checkpw(password, stored_hashed_password):
                return 'success'
        except ValueError:
            # Handle invalid salt error
            return 'fail'
    return 'fail'

# Function to get user information
def get_user(user_id):
    conn = sqlite3.connect(current_app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@auth.route('/login', methods=['POST'])
def login():
    logging.basicConfig(filename= 'authorization.log', level=logging.DEBUG)    
    # Get login credentials from request data
    logging.debug(f'hello world')
    data = request.json
    
    # Check if the data is empty
    if not data:
        logging.error('No data provided in the request body')
        return jsonify({'error': 'No data provided in the request body', 'status_code': 400}), 400

    username = data.get('username')
    password = data.get('password')

    logging.debug(f'username: {username}')
    logging.debug(f'password: {password}')

    # Validate credentials (you should replace this with your actual authentication logic)
    user = authenticate_user(username, password)
    
    logging.debug(f'user: {user}')

    if len(username) == 0 or len(password) == 0:
        return jsonify({'error': 'No data given', 'status_code': 400}), 400
    
    if user == 'success':
        # session['user_id'] = user[0]  # Assuming user[0] is the user's ID
        return jsonify({'message': 'Login successful', 'status_code': 200}), 200
    else:
        return jsonify({'message': 'Invalid username or password', 'status_code': 401}), 401
    
@auth.route('/register', methods=['POST'])
def register():
    # Get username and password from request data
    data = request.json

    # Check if the data is empty data sanitization
    if not data:
        logging.error('No data provided in the request body')
        return jsonify({'error': 'No data provided in the request body', 'status_code': 400}), 400
    
    username = data.get('username')
    password = data.get('password')

    #hash password then encrypts it
    hashed_password = hash_password(password)
    encrypted_hashed_password = encrypt_data(hashed_password)


    # Check if username or password is missing
    if not username or not password:
        return jsonify({'error': 'Username or password missing', 'status_code':400}), 400

    # Check if username already exists
    db = get_db()
    existing_user = db.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    if existing_user:
        return jsonify({'error': 'Username already exists', 'status_code': 400}), 400

    # Insert new user into the database
    db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, encrypted_hashed_password))
    db.commit()

    return jsonify({'message': 'User created successfully', 'status_code':201}), 201


@auth.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('auth.login'))

