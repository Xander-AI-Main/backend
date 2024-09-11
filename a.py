
import requests
import json

url = "http://127.0.0.1:8000/core/interference/" 
data = {
    "data": [37764, 25, 'Female', 'Europe', 'Action', 6.241870965810384, 0, 'Easy', 15, 160, 79, 45],
    "modelId": 'b13dde0a-07b0-4974-9e37-9676f33eb26c',
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