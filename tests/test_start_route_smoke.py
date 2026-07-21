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


class StartRouteSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_start_returns_lightweight_html(self):
        response = await call_asgi(main.app, "GET", "/start", headers={"host": "testserver"})
        body = bytes(response["body"])
        size = len(body)
        print(f"GET /start status={response['status']} bytes={size}")

        self.assertEqual(response["status"], 200)
        self.assertLess(size, 100_000)
        self.assertIn(b"dashboard.js", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
