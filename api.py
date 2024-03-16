from flask import Flask, Response, request, jsonify
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'OK'})

@app.route('/modelstealing')
def model():
    """_summary_

    Request:
        method=GET
        files={"file": image}
        headers={"token": TEAM_TOKEN}

    Response:
        _type_: encoded image
    """
    img_file = request.files["file"]
    img = Image.open(img_file)
    
    # model(img)

    return jsonify({'representation': 'OK'})


if __name__ == '__main__':
    app.run()
