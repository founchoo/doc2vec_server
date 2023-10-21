# Doc2Vec Server
A Python back-end server supporting to be invoked doc2vec method by other language via HTTP request.

# Doc2Vec method
I take the reference [here](https://juejin.cn/s/bert%20%E6%96%87%E6%9C%AC%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)

```
import torch
from transformers import BertTokenizer, BertModel

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 输入文本
text = "Hello, how are you?"

# 将文本转换为BERT需要的输入格式
input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0) 

# 使用BERT模型提取文本特征
outputs = model(input_ids)
word_embeddings = outputs.last_hidden_state  # 每个单词的词向量
sentence_embedding = outputs.pooler_output  # 整个句子的句子向量
```

# Python server
[Here](https://pythonbasics.org/webserver/) is the example.

```
# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

hostName = "localhost"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
```
