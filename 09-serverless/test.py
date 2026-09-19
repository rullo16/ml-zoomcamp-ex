import requests
import json

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

with open('customer.json', 'r') as f:
    customer = json.load(f)

result = requests.post(url, json=customer).json()

print(result)