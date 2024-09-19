
import requests
import json

url = "https://apiv3.xanderco.in/core/interference/" 

data = {
    "data": [0.0261, 0.0266, 0.0223, 0.0749, 0.1364, 0.1513, 0.1316, 0.1654, 0.1864, 0.2013, 0.289, 0.365, 0.351, 0.3495, 0.4325, 0.5398, 0.6237, 0.6876, 0.7329, 0.8107, 0.8396, 0.8632, 0.8747, 0.9607, 0.9716, 0.9121, 0.8576, 0.8798, 0.772, 0.5711, 0.4264, 0.286, 0.3114, 0.2066, 0.1165, 0.0185, 0.1302, 0.248, 0.1637, 0.1103, 0.2144, 0.2033, 0.1887, 0.137, 0.1376, 0.0307, 0.0373, 0.0606, 0.0399, 0.0169, 0.0135, 0.0222, 0.0175, 0.0127, 0.0022, 0.0124, 0.0054, 0.0021, 0.0028, 0.0023],
    "modelId": '95f1f60b-8acb-41df-bb97-4f0c48bb5d0b',
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
