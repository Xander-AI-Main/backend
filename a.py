
# import requests
# import json

# url = "https://api.xanderco.in/core/interference/" 

# data = {
#     "data": [16434, 31, 'Male', 'USA', 'Sports', 22.78055581220348, 1, 'Easy', 0, 178, 56, 27],
#     "modelId": '05c32688-925b-4abf-9211-db98d54b3294',
#     "userId": '24',
# }

# try:
#     response = requests.post(url, json=data)

#     if response.status_code == 200:
#         # Print the response JSON
#         print("Response:")
#         print(json.dumps(response.json(), indent=2))
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.text)
# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")


# import requests
# import json

# url = "https://api.xanderco.in/core/interference/" 

# data = {
#     "data": [0.0126, 0.0519, 0.0621, 0.0518, 0.1072, 0.2587, 0.2304, 0.2067, 0.3416, 0.4284, 0.3015, 0.1207, 0.3299, 0.5707, 0.6962, 0.9751, 1.0, 0.9293, 0.621, 0.4586, 0.5001, 0.5032, 0.7082, 0.842, 0.8109, 0.769, 0.8105, 0.6203, 0.2356, 0.2595, 0.6299, 0.6762, 0.2903, 0.4393, 0.8529, 0.718, 0.4801, 0.5856, 0.4993, 0.2866, 0.0601, 0.1167, 0.2737, 0.2812, 0.2078, 0.066, 0.0491, 0.0345, 0.0172, 0.0287, 0.0027, 0.0208, 0.0048, 0.0199, 0.0126, 0.0022, 0.0037, 0.0034, 0.0114, 0.0077],
#     "modelId": '9b60865e-87d6-4cff-8a1d-53bc097ce6b7',
#     "userId": '24',
# }

# try:
#     response = requests.post(url, json=data)

#     if response.status_code == 200:
#         # Print the response JSON
#         print("Response:")
#         print(json.dumps(response.json(), indent=2))
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.text)
# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")



import requests
import json

url = "https://api.xanderco.in/core/interference/" 

data = {
    "data": "Buy moreee",
    "modelId": '6ff003a4-0286-4a7a-9fdd-bb5ffaaa857a',
    "userId": '22',
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
