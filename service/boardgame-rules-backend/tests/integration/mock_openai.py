import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    EMBEDDING_DIM = int(os.getenv("MOCK_EMBEDDING_DIM", "768"))

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}

        if self.path.endswith("/v1/embeddings"):
            data = payload.get("input", "")
            if isinstance(data, list):
                size = len(data)
            else:
                size = 1
            embedding = [0.001] * self.EMBEDDING_DIM
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": embedding,
                            "index": idx,
                        }
                        for idx in range(size)
                    ],
                    "model": payload.get("model", "mock-embedding"),
                }
            )
            return

        if self.path.endswith("/v1/chat/completions"):
            context_detected = False
            for msg in payload.get("messages", []):
                content = msg.get("content") or ""
                if "10 очков победы" in content:
                    context_detected = True
                    break
            answer = (
                "CONTEXT_OK: в правилах есть условие победы 10 очков."
                if context_detected
                else "NO_CONTEXT: контекст правил не найден."
            )
            self._send_json(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload.get("model", "mock-llm"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": answer,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            return

        self._send_json({"detail": "Not found"}, status=404)

    def log_message(self, format: str, *args):  # noqa: A003
        return


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    server.serve_forever()
