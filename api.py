from flask import Flask, Response, request, jsonify
import torch
import clip
from PIL import Image

class ModelApi:    

    def __init__(self):
        self.app = Flask(__name__)
        self.model, self.preprocess = self.setup_model()
        
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/modelstealing', 'modelstealing', self.modelstealing)

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
        img = self.preprocess(Image.open(img_file)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(img)

            # Add noise to the image features
            noise = torch.randn_like(image_features) * 0.01
            noisy_image_features = image_features + noise

        image_features_list = noisy_image_features.tolist()

        return jsonify({'representation': image_features_list})


if __name__ == '__main__':
    model = ModelApi()
    model.app.run()