import asyncio

import main


async def call(app, method, path):
    response = {"status": None, "body": bytearray()}
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
        elif message["type"] == "http.response.body":
            response["body"].extend(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }

    await app(scope, receive, send)
    return response


async def main_async():
    for path in ["/", "/start", "/docs"]:
        resp = await call(main.app, "GET", path)
        print(f"{path} status={resp['status']} bytes={len(resp['body'])}")


if __name__ == "__main__":
    asyncio.run(main_async())
