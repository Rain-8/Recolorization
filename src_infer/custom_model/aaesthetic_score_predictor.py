import json
from pyexpat import model
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1

#
# Load the aesthetics predictor
#
def load_model():
    model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
    model = AestheticsPredictorV1.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def predict_aesthetic_score(file):
    model, processor = load_model()
    data = json.load(open(file))
    for key, value in data.items():
        # image_path = value['tgt_image_path']
        image_path = value['pal_image_path']
        image = Image.open(image_path).resize((256, 256)).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = outputs.logits
        print(f"Aesthetics score for {key}: {prediction}")

if __name__ == "__main__":
    predict_aesthetic_score("Test_results/test_meta.json")