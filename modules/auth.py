from functools import wraps
import sqlite3
from flask import Blueprint, request, session, redirect, url_for, request, g, jsonify, current_app
import logging
import json
from modules.database import get_db


auth = Blueprint('auth', __name__)

auth.config = {}

logging.basicConfig(filename='auth.log', level=logging.DEBUG)

# Function to authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect(current_app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

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

@auth.route('/')
def index():
    return 'Welcome to the Home Page'

@auth.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        g.user = get_user(session['user_id'])

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
    
    if user:
        # session['user_id'] = user[0]  # Assuming user[0] is the user's ID
        return jsonify({'message': 'Login successful', 'status_code': 200}), 200
    else:
        return jsonify({'message': 'Invalid username or password', 'status_code': 401}), 401
    
@auth.route('/register', methods=['POST'])
def register():
    # Get username and password from request data
    data = request.json

    # Check if the data is empty
    if not data:
        logging.error('No data provided in the request body')
        return jsonify({'error': 'No data provided in the request body', 'status_code': 400}), 400
    
    username = data.get('username')
    password = data.get('password')

    # Check if username or password is missing
    if not username or not password:
        return jsonify({'error': 'Username or password missing', 'status_code':400}), 400

    # Check if username already exists
    db = get_db()
    existing_user = db.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    if existing_user:
        return jsonify({'error': 'Username already exists', 'status_code': 400}), 400

    # Insert new user into the database
    db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    db.commit()

    return jsonify({'message': 'User created successfully', 'status_code':201}), 201


# @auth.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         # Validate login credentials
#         username = request.form['username']
#         password = request.form['password']

#         logging.debug(f'username: {username}')
#         logging.debug(f'password: {password}')
        
#         user = authenticate_user(username, password)
#         if user:
#             session['user_id'] = user[0]  # Assuming user[0] is the user's ID
#             return redirect(url_for('auth.protected'))
#         else:
#             error = 'Invalid username or password'
#             return render_template('login.html', error=error)
#     return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('auth.login'))

@auth.route('/protected')
@login_required
def protected():
    return 'This is a protected page'