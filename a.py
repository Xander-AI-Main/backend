
import requests
import json

url = "https://apiv2.xanderco.in/core/interference/" 

data = {
    "data": [0.0262, 0.0582, 0.1099, 0.1083, 0.0974, 0.228, 0.2431, 0.3771, 0.5598, 0.6194, 0.6333, 0.706, 0.5544, 0.532, 0.6479, 0.6931, 0.6759, 0.7551, 0.8929, 0.8619, 0.7974, 0.6737, 0.4293, 0.3648, 0.5331, 0.2413, 0.507, 0.8533, 0.6036, 0.8514, 0.8512, 0.5045, 0.1862, 0.2709, 0.4232, 0.3043, 0.6116, 0.6756, 0.5375, 0.4719, 0.4647, 0.2587, 0.2129, 0.2222, 0.2111, 0.0176, 0.1348, 0.0744, 0.013, 0.0106, 0.0033, 0.0232, 0.0166, 0.0095, 0.018, 0.0244, 0.0316, 0.0164, 0.0095, 0.0078],
    "modelId": '5edc21f9-686e-420b-8b14-def1bc771a0a',
    "userId": '41',
}

try:
    response = requests.post(url, json=data)

    if response.status_code == 200:
        # Print the response JSON
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
