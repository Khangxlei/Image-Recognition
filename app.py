from flask import Flask

from modules.database import *
from modules.upload_images import *
from modules.upload_labels import *
from modules.train_model import *

app = Flask(__name__)

app.register_blueprint(upload_images)
app.register_blueprint(upload_labels)
app.register_blueprint(train_model)

app.config['DATABASE'] = 'database.db'
    
    
if __name__ == '__main__':
    init_db(app)
    app.teardown_appcontext(close_connection)
    app.run(debug=True)
