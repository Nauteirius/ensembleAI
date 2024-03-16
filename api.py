from flask import Flask, Response, request, jsonify
import torch
import clip
from PIL import Image

class ModelApi:    

    def __init__(self):
        self.app = Flask(__name__)
        self.model, self.preprocess = self.setup_model()
        
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/', 'modelstealing', self.modelstealing)

    def setup_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def index(self):
        is_initialized = (self.model is not None)
        return jsonify({'model_initialized': is_initialized, "device": self.device})

    def modelstealing(self):
        """_summary_

        Request:
            method=GET
            files={"file": image}
            headers={"token": TEAM_TOKEN}

        Response:
            _type_: encoded image
        """
        img_file = request.files["file"]
        img = Image.open(img_file).to(self.device)
        
        with torch.no_grad():
            image_features = model.encode_image(img)

        return jsonify({'representation': image_features})


if __name__ == '__main__':
    model = ModelApi()