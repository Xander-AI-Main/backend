
import requests
import json

url = "http://127.0.0.1:8000/core/interference/" 

data = {
    "data": [2325.0, 15.0, 1.0, 0.0, 1.0, 8.489856653530703, 25.0, 1.0, 4.0, 1.0, 0.0, 0.0, 0.0, 1.283227506286419],
    "modelId": '861baeff-9e9f-4f10-bd37-70d853fc52d2',
    "userId": '40',
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