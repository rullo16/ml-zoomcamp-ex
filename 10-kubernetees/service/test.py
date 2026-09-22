import requests

url = "http://127.0.0.1:30080/predict"

request = {
    "url": "http://bit.ly/mlbookcamp-pants"
}

response = requests.post(url, json=request)
result = response.json()

print(f"Top prediction: {result['top_class']} ({result['top_probability']:.2f})")
print(f"\nAll predictions:")
for cls, prob in result['prediction'].items():
    print(f"{cls:12s}: {prob:.2%}")