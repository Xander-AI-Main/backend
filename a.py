
import requests
import json

url = "http://127.0.0.1:8000/core/interference/" 

data = {
    "data": [62257, 'Chevrolet', 'Camaro 1SS', 2017, 50850, 'Gasoline', '455.0HP 6.2L 8 Cylinder Engine Gasoline Fuel', 'A/T', 'Orange', 'Black', 'None reported'],
    "modelId": '210a8cf6-9132-483a-88be-3748fb1f3468',
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
