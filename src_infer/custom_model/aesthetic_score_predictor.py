import json
from pyexpat import model
from unittest import result
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
    result = {}
    for key, value in data.items():
        result[key] = {}
        tgt_image_path = value['tgt_image_path']
        pal_image_path = value['pal_image_path']
        src_image_path = key
        image = Image.open(tgt_image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = outputs.logits.item()
        result[key]["tgt_aesthetics_score"] = prediction

        image = Image.open(pal_image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = outputs.logits.item()
        result[key]["pal_aesthetics_score"] = prediction

        image = Image.open(src_image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = outputs.logits.item()
        result[key]["src_aesthetics_score"] = prediction
    file = "test_results/aesthetic_scores.json"
    json.dump(result, open(file, 'w'))

if __name__ == "__main__":
    predict_aesthetic_score("test_results/test_meta.json")