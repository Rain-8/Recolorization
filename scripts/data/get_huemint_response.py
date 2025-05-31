import requests
import json


def get_huemint_response(adjacency_matrix, num_colors):
    json_data = {
        "mode": "transformer",  # transformer, diffusion, or random
        "num_colors": num_colors,        # max 12, min 2
        "temperature": "1.8",   # max 2.4, min 0
        "num_results": 140,      # max 50 for transformer, 5 for diffusion
        "adjacency": adjacency_matrix
    }   
    url = "https://api.huemint.com/color"
    response = requests.post(url, data=json.dumps(json_data), headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        response_data = response.json()
        return response_data
    return None