
import requests
import json

url = "https://api.xanderco.in/core/interference/" 

data = {
    "data": [2.0, 2.0, 29.0, 1.0, 3.0, 44.0, 2.0, 9.0, 0.0, 1.0, 6.0],
    "modelId": '78467fa5-d6d6-4f0e-b466-9fb067bb000a',
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
