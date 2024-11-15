from flask import Flask
from next_question.app1 import app as app1
from question_pred.app2 import app as app2

app = Flask(__name__)

app.register_blueprint(app1, url_prefix='/next')
app.register_blueprint(app2, url_prefix='/pred')

if __name__ == "__main__":
    app.run()
