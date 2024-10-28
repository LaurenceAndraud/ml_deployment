import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "feature1": 5.1,
    "feature2": 3.5,
    "feature3": 1.4,
    "feature4": 0.2
}

response = requests.post(url, json=data)
print(response.json())