
import requests
import json

url = "http://127.0.0.1:8000/core/interference/" 

data = {
    "data": [0.034, 0.0625, 0.0381, 0.0257, 0.0441, 0.1027, 0.1287, 0.185, 0.2647, 0.4117, 0.5245, 0.5341, 0.5554, 0.3915, 0.295, 0.3075, 0.3021, 0.2719, 0.5443, 0.7932, 0.8751, 0.8667, 0.7107, 0.6911, 0.7287, 0.8792, 1.0, 0.9816, 0.8984, 0.6048, 0.4934, 0.5371, 0.4586, 0.2908, 0.0774, 0.2249, 0.1602, 0.3958, 0.6117, 0.5196, 0.2321, 0.437, 0.3797, 0.4322, 0.4892, 0.1901, 0.094, 0.1364, 0.0906, 0.0144, 0.0329, 0.0141, 0.0019, 0.0067, 0.0099, 0.0042, 0.0057, 0.0051, 0.0033, 0.0058],
    "modelId": '7d25570f-ae7c-44a6-91ae-6c7151b9dce5',
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