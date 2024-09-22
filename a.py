import requests
import json

url = "http://127.0.0.1:8000/core/interference/"

data = {
    "data": "Explain the innovation introduced in working of swarm robots?",
    "modelId": "d19ce71e-c7cc-4ced-89df-4e676a52a267",
    "userId": "48",
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
