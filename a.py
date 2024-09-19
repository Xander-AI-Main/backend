
import requests
import json

url = "https://apiv3.xanderco.in/core/interference/" 

data = {
    "data": [44, 'male', 21.85, 3, 'no', 'northeast'],
    "modelId": '1e3dced2-4d29-49ab-8a33-66c164bf42bf',
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
