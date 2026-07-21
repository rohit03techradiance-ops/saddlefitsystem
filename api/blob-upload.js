const { handleUpload } = require("@vercel/blob/client");

const DEFAULT_MAX_BYTES = 200 * 1024 * 1024;
const TOKEN_TTL_MS = 15 * 60 * 1000;
const ALLOWED_CONTENT_TYPES = ["video/mp4", "video/quicktime"];

function parseMaxBytes(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_MAX_BYTES;
  }
  return Math.max(1, Math.floor(parsed));
}

const MAX_BYTES = parseMaxBytes(process.env.VIDEO_UPLOAD_MAX_BYTES || DEFAULT_MAX_BYTES);

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
      if (raw.length > 128 * 1024) {
        reject(new Error("Request body too large."));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (!raw) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch (_error) {
        reject(new Error("Invalid JSON request body."));
      }
    });
    req.on("error", reject);
  });
}

function validateUploadPathname(pathname) {
  const normalized = String(pathname || "").replace(/\\/g, "/").replace(/^\/+/, "");
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length < 3 || parts[0] !== "videos") {
    throw new Error("Invalid blob upload path.");
  }
  if (parts[1] === "analysis") {
    if (parts.length !== 3) {
      throw new Error("Invalid analysis upload path.");
    }
  } else if (parts[1] === "comparison") {
    if (parts.length !== 4 || !["a", "b"].includes(parts[2])) {
      throw new Error("Invalid comparison upload path.");
    }
  } else {
    throw new Error("Invalid blob upload scope.");
  }
  const filename = parts[parts.length - 1];
  if (!/\.(mp4|mov)$/i.test(filename)) {
    throw new Error("Only MP4 and MOV videos are allowed.");
  }
  return normalized;
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.statusCode = 405;
    res.setHeader("Allow", "POST");
    res.end("Method Not Allowed");
    return;
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch (error) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: error.message || "Invalid upload authorization request." }));
    return;
  }

  try {
    const jsonResponse = await handleUpload({
      body,
      request: req,
      token: process.env.BLOB_READ_WRITE_TOKEN,
      onBeforeGenerateToken: async (pathname, clientPayload, multipart) => {
        const safePathname = validateUploadPathname(pathname);
        return {
          allowedContentTypes: ALLOWED_CONTENT_TYPES,
          maximumSizeInBytes: MAX_BYTES,
          validUntil: new Date(Date.now() + TOKEN_TTL_MS),
          addRandomSuffix: false,
          allowOverwrite: false,
          tokenPayload: JSON.stringify({
            pathname: safePathname,
            clientPayload: clientPayload || "",
            multipart: !!multipart,
          }),
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        console.log("blob upload completed", blob.pathname, tokenPayload || "");
      },
    });

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(jsonResponse));
  } catch (error) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: error.message || "Upload authorization failed." }));
  }
};
