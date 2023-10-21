from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from transformers import BertTokenizer, BertModel

hostName = "localhost"
serverPort = 8080
model = None

class Doc2Vec:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def getVector(self, text: str) -> str:
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        return str(outputs.pooler_output.detach().numpy()[0])

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        if (self.path.startswith('/text=')):
            text = self.path[6:]
            self.wfile.write(bytes(model.getVector(text), "utf-8"))

if __name__ == "__main__":
    model = Doc2Vec()
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")