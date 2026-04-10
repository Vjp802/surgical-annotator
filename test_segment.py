import requests

res = requests.post("http://127.0.0.1:8000/api/segment/image", json={
    "image_id": "8316dac814a549149d14ad6ba3ce7950",
    "x": 100,
    "y": 100
})
print(res.status_code)
print(res.text)
