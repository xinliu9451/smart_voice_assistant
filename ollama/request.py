import requests

# 生成文本
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "prompt": "你好，你能帮我写一段代码吗？",
        "stream": False
    }
)
print(response.json()['response'])