
import requests
import json

url = "http://127.0.0.1:8000/core/interference/" 

data = {
    "data": "Your input text",
    "modelId": '15393571-06d9-4224-b834-685bae916110',
    "userId": '48',
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