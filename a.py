
import requests
import json

url = "https://api.xanderco.in/core/interference/" 

data = {
    "data": "What is munafa",
    "modelId": '4424199c-ff37-43a7-9ebb-43a87251a1d3',
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
