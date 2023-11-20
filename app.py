from flask import Flask, request
from flask_cors import CORS
from ai import get_ai_move, get_model

app = Flask(__name__)
cors = CORS(app, resources={r"/getmove": {"origins": "https://main--unique-toffee-499537.netlify.app"}})

model = get_model('model1.keras')

@app.route('/getmove', methods=['POST'])
def get_move():
    request_data = request.get_json()
    move = get_ai_move(request_data['fen'], model)
    return move

@app.errorhandler(404)
def page_not_found(e):
    return '404 page not found!'

if __name__ == '__main__':
    app.run(port=8010)