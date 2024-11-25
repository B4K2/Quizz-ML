from flask import Flask
from next_question.app1 import app as app1
from question_pred.app2 import app as app2
from summary.app3 import app as app3
from summary.app4 import app as app4

app = Flask(__name__)

app.register_blueprint(app1, url_prefix='/next')
app.register_blueprint(app2, url_prefix='/pred')
app.register_blueprint(app3, url_prefix='/sum')
app.register_blueprint(app4, url_prefix='/summ')

if __name__ == "__main__":
    app.run(debug=True)
