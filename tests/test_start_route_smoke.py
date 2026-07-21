import asyncio
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


async def call_asgi(app, method: str, path: str, body: bytes = b"", headers=None):
    headers = headers or {}
    response = {"status": None, "headers": [], "body": bytearray()}
    request_body_sent = False

    async def receive():
        nonlocal request_body_sent
        if request_body_sent:
            return {"type": "http.disconnect"}
        request_body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"].extend(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }

    await app(scope, receive, send)
    return response


def header_value(response, name: str) -> str:
    name_bytes = name.lower().encode("latin-1")
    for key, value in response["headers"]:
        if key == name_bytes:
            return value.decode("latin-1")
    return ""


class StartRouteSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_start_returns_lightweight_html(self):
        paths = ["/", "/start", "/docs", "/dashboard.js", "/favicon.svg", "/favicon.ico", "/favicon.png"]
        for path in paths:
            with self.subTest(path=path):
                response = await call_asgi(main.app, "GET", path, headers={"host": "testserver"})
                body = bytes(response["body"])
                size = len(body)
                content_type = header_value(response, "content-type")
                print(f"GET {path} status={response['status']} bytes={size} content_type={content_type}")

                self.assertEqual(response["status"], 200)
                self.assertGreater(size, 0)
                self.assertLess(size, 100_000)

                if path in {"/", "/start"}:
                    self.assertIn(b"dashboard.js", body)
                elif path == "/docs":
                    self.assertIn("text/html", content_type)
                    self.assertIn(b"<html", body)
                elif path == "/dashboard.js":
                    self.assertIn(b"window", body)
                else:
                    self.assertIn("image/svg+xml", content_type)
                    self.assertIn(b"<svg", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
