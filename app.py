from flask import Flask
import os

from modules.database import *
from modules.upload_data import *
from modules.train_model import *
from modules.upload_models import *
from modules.inference_data import *
from modules.auth import *

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

app.register_blueprint(auth)
app.register_blueprint(upload_images)
app.register_blueprint(upload_labels)
app.register_blueprint(upload_models)
app.register_blueprint(train_model)
app.register_blueprint(inference_data)

app.config['DATABASE'] = 'database.db'

app.secret_key = os.urandom(16)

if __name__ == '__main__':
    db_path = app.config['DATABASE']
    if not os.path.exists(db_path):
        init_db(app)
    app.teardown_appcontext(close_connection)
    app.run(debug=True)