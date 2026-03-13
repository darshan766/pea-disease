import requests

url = "http://127.0.0.1:5000/predict"

files = {"image": open(r"C:\Users\darsh\Desktop\Pea_disease\dataset\Test\FRESH_LEAF_1\resized_149.jpg", "rb")}

response = requests.post(url, files=files)

print(response.json())