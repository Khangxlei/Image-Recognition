from flask import Flask

from modules.database import *
from modules.upload_images import *
from modules.upload_labels import *
from modules.train_model import *
from modules.upload_models import *

app = Flask(__name__)

app.register_blueprint(upload_images)
app.register_blueprint(upload_labels)
app.register_blueprint(train_model)
app.register_blueprint(upload_models)

app.config['DATABASE'] = 'database.db'
    
    
if __name__ == '__main__':
    db_path = app.config['DATABASE']
    if not os.path.exists(db_path):
        init_db(app)
    app.teardown_appcontext(close_connection)
    app.run(debug=True)

    # NEXT STEPS: work on attaching a user id and dataset id every time user uploads the data, when the user 
    # wants to train their model, they specify the dataset id that they want to train with. After that, there will be 
    # a model ID that the user specify to train as well. Once we get all of this, we should be able to allow user 
    # to pick whichever model and whichever dataset to train and test on. 
    
    # work on data analysis BEFORE training. 
    # Should also work on how to allow users to use certain datasets

