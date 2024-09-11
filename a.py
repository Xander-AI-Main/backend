
import requests
import json

url = "http://127.0.0.1:8000/core/interference/" 
data = {
    "data": [32, 'male', 28.12, 4, 'no', 'northwest'],
    "modelId": '1c42f620-3d3f-4c26-a0a6-356cbac35c73',
    "userId": '55',
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