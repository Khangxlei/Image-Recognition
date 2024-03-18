from flask import Flask

from modules.database import *
from modules.upload_images import *
from modules.upload_labels import *
from modules.train_model import *
from modules.upload_models import *
from modules.inference_data import *

app = Flask(__name__)

app.register_blueprint(upload_images)
app.register_blueprint(upload_labels)
app.register_blueprint(train_model)
app.register_blueprint(inference_data)
app.register_blueprint(upload_models)

app.config['DATABASE'] = 'database.db'
    
    
if __name__ == '__main__':
    db_path = app.config['DATABASE']
    if not os.path.exists(db_path):
        init_db(app)
    app.teardown_appcontext(close_connection)
    app.run(debug=True)

