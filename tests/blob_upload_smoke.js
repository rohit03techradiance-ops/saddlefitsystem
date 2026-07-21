const assert = require("node:assert/strict");
const { createBlobUploadHandler } = require("../api/blob-upload");


function createRequest(body, headers = {}, method = "POST") {
  return {
    method,
    headers: {
      "content-type": "application/json",
      ...headers,
    },
    body,
  };
}


function createResponse() {
  return {
    statusCode: null,
    headers: {},
    body: "",
    setHeader(name, value) {
      this.headers[String(name).toLowerCase()] = value;
    },
    end(chunk) {
      if (typeof chunk === "string") {
        this.body += chunk;
      } else if (Buffer.isBuffer(chunk)) {
        this.body += chunk.toString("utf8");
      } else if (chunk != null) {
        this.body += String(chunk);
      }
    },
  };
}


async function runCase(name, handler, request) {
  const response = createResponse();
  await handler(request, response);
  const parsed = response.body ? JSON.parse(response.body) : {};
  console.log(`${name} status=${response.statusCode} body=${response.body}`);
  return { response, parsed };
}


async function main() {
  const invalidRequestHandler = createBlobUploadHandler({
    env: { BLOB_READ_WRITE_TOKEN: "test-token", VIDEO_UPLOAD_MAX_BYTES: "1048576" },
    logger: { info() {}, error() {} },
    handleUpload: async () => {
      throw new Error("handleUpload should not run for invalid metadata");
    },
  });

  const invalidResult = await runCase(
    "invalid-request",
    invalidRequestHandler,
    createRequest({}),
  );
  assert.equal(invalidResult.response.statusCode, 400);
  assert.equal(invalidResult.parsed.success, false);
  assert.match(invalidResult.parsed.error, /Missing pathname/i);

  const missingEnvHandler = createBlobUploadHandler({
    env: {},
    logger: { info() {}, error() {} },
    handleUpload: async () => {
      throw new Error("handleUpload should not run when blob token is missing");
    },
  });

  const missingEnvResult = await runCase(
    "missing-env",
    missingEnvHandler,
    createRequest({
      pathname: "videos/analysis/abc123-rider.webm",
      clientPayload: JSON.stringify({
        originalFilename: "rider.webm",
        contentType: "video/webm",
        scope: "analysis",
      }),
      multipart: true,
    }),
  );
  assert.equal(missingEnvResult.response.statusCode, 503);
  assert.equal(missingEnvResult.parsed.success, false);
  assert.match(missingEnvResult.parsed.error, /Blob storage is not configured/i);

  let capturedOptions = null;
  const validRequestHandler = createBlobUploadHandler({
    env: {
      BLOB_READ_WRITE_TOKEN: "test-token",
      VIDEO_UPLOAD_MAX_BYTES: "1048576",
      VERCEL_BLOB_CALLBACK_URL: "https://example.ngrok-free.app/api/blob-upload",
    },
    logger: { info() {}, error() {} },
    handleUpload: async (options) => {
      capturedOptions = options;
      const tokenConfig = await options.onBeforeGenerateToken(
        "videos/analysis/abc123-rider.webm",
        JSON.stringify({
          originalFilename: "rider.webm",
          contentType: "video/webm",
          scope: "analysis",
        }),
        true,
      );
      assert(tokenConfig.allowedContentTypes.includes("video/webm"));
      assert.equal(tokenConfig.maximumSizeInBytes, 1048576);
      assert.equal(tokenConfig.addRandomSuffix, false);
      assert.equal(tokenConfig.allowOverwrite, false);
      assert.equal(tokenConfig.callbackUrl, "https://example.ngrok-free.app/api/blob-upload");
      const payload = JSON.parse(tokenConfig.tokenPayload);
      assert.equal(payload.originalFilename, "rider.webm");
      assert.equal(payload.contentType, "video/webm");
      assert.equal(payload.scope, "analysis");
      assert.equal(payload.multipart, true);
      return {
        type: "blob.generate-client-token",
        clientToken: "token-123",
      };
    },
  });

  const validResult = await runCase(
    "valid-metadata",
    validRequestHandler,
    createRequest({
      pathname: "videos/analysis/abc123-rider.webm",
      clientPayload: JSON.stringify({
        originalFilename: "rider.webm",
        contentType: "video/webm",
        scope: "analysis",
      }),
      multipart: true,
    }),
  );
  assert.equal(validResult.response.statusCode, 200);
  assert.deepEqual(validResult.parsed, {
    type: "blob.generate-client-token",
    clientToken: "token-123",
  });
  assert(capturedOptions, "handleUpload should have been called");
  assert.equal(capturedOptions.token, "test-token");
  assert.equal(capturedOptions.body.pathname, "videos/analysis/abc123-rider.webm");
  assert.equal(capturedOptions.request.method, "POST");

  console.log("blob upload smoke test complete");
}


main().catch((error) => {
  console.error(error);
  process.exit(1);
});
