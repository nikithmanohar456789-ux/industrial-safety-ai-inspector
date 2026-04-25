import requests, json

query = input("Enter your query here: ")

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:4b",
        "prompt": query,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line.decode("utf-8"))
        print(chunk.get("response", ""), end="", flush=True)