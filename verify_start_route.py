import asyncio

import main


async def call(app, method, path):
    response = {"status": None, "headers": [], "body": bytearray()}
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
            response["headers"] = message.get("headers", [])
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


def header_value(response, name):
    name_bytes = name.lower().encode("latin-1")
    for key, value in response["headers"]:
        if key == name_bytes:
            return value.decode("latin-1")
    return ""


async def main_async():
    for path in ["/", "/start", "/docs", "/dashboard.js", "/favicon.svg", "/favicon.ico", "/favicon.png"]:
        resp = await call(main.app, "GET", path)
        print(
            f"{path} status={resp['status']} bytes={len(resp['body'])} "
            f"content_type={header_value(resp, 'content-type')}"
        )


if __name__ == "__main__":
    asyncio.run(main_async())
