const ALLOWED_CONTENT_TYPES = ["video/mp4", "video/quicktime", "video/webm"];
const DEFAULT_MAX_BYTES = 200 * 1024 * 1024;
const TOKEN_TTL_MS = 15 * 60 * 1000;
const UPLOAD_PREFIX = "videos";
const EXPECTED_SCHEMA = {
  pathname: "string",
  clientPayload: "optional JSON string",
  multipart: "optional boolean",
};

function parseMaxBytes(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_MAX_BYTES;
  }
  return Math.max(1, Math.floor(parsed));
}

function loadHandleUpload() {
  try {
    return require("@vercel/blob/client").handleUpload;
  } catch (_error) {
    return null;
  }
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function normalizeString(value) {
  return String(value || "").trim();
}

function normalizeContentType(value) {
  return normalizeString(value).split(";", 1)[0].toLowerCase();
}

function parseClientPayload(clientPayload) {
  if (typeof clientPayload !== "string" || !clientPayload.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(clientPayload);
    return isPlainObject(parsed) ? parsed : {};
  } catch (_error) {
    return {};
  }
}

function inspectBlobPathname(pathname) {
  const normalized = normalizeString(pathname).replace(/\\/g, "/").replace(/^\/+/, "");
  const parts = normalized.split("/").filter(Boolean);
  const filename = parts[parts.length - 1] || "";
  const extension = filename.includes(".") ? `.${filename.split(".").pop().toLowerCase()}` : "";
  return {
    normalized,
    parts,
    filename,
    extension,
    scope: parts[1] || "",
    slot: parts[2] || "",
  };
}

function validateUploadPathname(pathname) {
  const info = inspectBlobPathname(pathname);
  if (!info.normalized) {
    throw new Error("Missing blob pathname.");
  }
  if (info.parts.length < 3 || info.parts[0] !== UPLOAD_PREFIX) {
    throw new Error("Invalid blob upload path.");
  }
  if (info.parts[1] === "analysis") {
    if (info.parts.length !== 3) {
      throw new Error("Invalid analysis upload path.");
    }
  } else if (info.parts[1] === "comparison") {
    if (info.parts.length !== 4 || !["a", "b"].includes(info.parts[2])) {
      throw new Error("Invalid comparison upload path.");
    }
  } else {
    throw new Error("Invalid blob upload scope.");
  }
  if (!/\.(mp4|mov|webm)$/i.test(info.filename)) {
    throw new Error("Only MP4, MOV, and WEBM videos are allowed.");
  }
  return info.normalized;
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    if (isPlainObject(req.body)) {
      resolve(req.body);
      return;
    }
    if (typeof req.body === "string") {
      try {
        resolve(req.body ? JSON.parse(req.body) : {});
      } catch (_error) {
        reject(new Error("Invalid JSON request body."));
      }
      return;
    }
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

function writeJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(payload));
}

function logDiagnostics(logger, level, message, details) {
  const sink = logger && typeof logger[level] === "function" ? logger[level].bind(logger) : console[level].bind(console);
  sink(message, details);
}

function createBlobUploadHandler(options = {}) {
  const env = options.env || process.env;
  const logger = options.logger || console;
  const now = options.now || Date.now;
  const handleUpload = options.handleUpload || loadHandleUpload();
  const maxBytes = parseMaxBytes(env.VIDEO_UPLOAD_MAX_BYTES || DEFAULT_MAX_BYTES);
  const callbackUrl = normalizeString(env.VERCEL_BLOB_CALLBACK_URL);

  return async function handler(req, res) {
    if (normalizeString(req.method).toUpperCase() !== "POST") {
      res.setHeader("Allow", "POST");
      writeJson(res, 405, {
        success: false,
        error: "Method Not Allowed",
      });
      return;
    }

    let body;
    try {
      body = await readJsonBody(req);
    } catch (error) {
      writeJson(res, 400, {
        success: false,
        error: error.message || "Invalid upload authorization request.",
        expected_schema: EXPECTED_SCHEMA,
      });
      return;
    }

    const requestContentType = normalizeString(req.headers && req.headers["content-type"]);
    const receivedFields = Object.keys(isPlainObject(body) ? body : {});
    const pathname = normalizeString(body && body.pathname);
    const clientPayload = parseClientPayload(body && body.clientPayload);
    const pathInfo = inspectBlobPathname(pathname);
    const declaredContentType = normalizeContentType(
      clientPayload.contentType || body.contentType || body.fileContentType || body.mimeType,
    );
    const filename = normalizeString(
      clientPayload.originalFilename || body.filename || pathInfo.filename || "",
    );

    logDiagnostics(logger, "info", "[blob-upload] request", {
      requestContentType,
      expectedSchema: EXPECTED_SCHEMA,
      receivedFields,
      pathname: pathInfo.normalized || null,
      filename: filename || null,
      fileContentType: declaredContentType || null,
      hasBLOB_READ_WRITE_TOKEN: Boolean(env.BLOB_READ_WRITE_TOKEN),
      hasVERCEL_BLOB_CALLBACK_URL: Boolean(callbackUrl),
    });

    if (!env.BLOB_READ_WRITE_TOKEN) {
      writeJson(res, 503, {
        success: false,
        error: "Blob storage is not configured.",
        hint: "Add BLOB_READ_WRITE_TOKEN in Vercel Project Settings or connect a Vercel Blob store.",
        required_env: ["BLOB_READ_WRITE_TOKEN"],
      });
      return;
    }

    if (!pathname) {
      writeJson(res, 400, {
        success: false,
        error: "Missing pathname.",
        expected_schema: EXPECTED_SCHEMA,
      });
      return;
    }

    try {
      validateUploadPathname(pathname);
    } catch (error) {
      writeJson(res, 415, {
        success: false,
        error: error.message || "Unsupported upload path.",
        expected_schema: EXPECTED_SCHEMA,
      });
      return;
    }

    if (declaredContentType && !ALLOWED_CONTENT_TYPES.includes(declaredContentType)) {
      writeJson(res, 415, {
        success: false,
        error: "Unsupported video format.",
        allowed_content_types: ALLOWED_CONTENT_TYPES,
      });
      return;
    }

    if (!handleUpload) {
      writeJson(res, 503, {
        success: false,
        error: "Blob upload SDK is unavailable.",
        hint: "Install @vercel/blob so the handleUpload helper is available at runtime.",
      });
      return;
    }

    try {
      const jsonResponse = await handleUpload({
        body,
        request: req,
        token: env.BLOB_READ_WRITE_TOKEN,
        onBeforeGenerateToken: async (path, clientPayloadString, multipart) => {
          const safeTokenPathname = validateUploadPathname(path);
          const metadata = parseClientPayload(clientPayloadString);
          const metadataContentType = normalizeContentType(metadata.contentType || declaredContentType);
          if (metadataContentType && !ALLOWED_CONTENT_TYPES.includes(metadataContentType)) {
            throw new Error("Unsupported video format.");
          }

          const tokenPayload = {
            pathname: safeTokenPathname,
            originalFilename: normalizeString(metadata.originalFilename || filename || pathInfo.filename || ""),
            contentType: metadataContentType || "",
            scope: normalizeString(metadata.scope || pathInfo.scope || ""),
            slot: normalizeString(metadata.slot || pathInfo.slot || ""),
            multipart: Boolean(multipart),
            maxBytes,
          };

          return {
            allowedContentTypes: ALLOWED_CONTENT_TYPES,
            maximumSizeInBytes: maxBytes,
            validUntil: new Date(now() + TOKEN_TTL_MS),
            addRandomSuffix: false,
            allowOverwrite: false,
            ...(callbackUrl ? { callbackUrl } : {}),
            tokenPayload: JSON.stringify(tokenPayload),
          };
        },
        onUploadCompleted: async ({ blob, tokenPayload }) => {
          logDiagnostics(logger, "info", "[blob-upload] completed", {
            pathname: blob && blob.pathname ? blob.pathname : null,
            url: blob && blob.url ? blob.url : null,
            contentType: blob && blob.contentType ? blob.contentType : null,
            tokenPayload: tokenPayload || null,
          });
        },
      });

      writeJson(res, 200, jsonResponse);
    } catch (error) {
      logDiagnostics(logger, "error", "[blob-upload] failed", {
        message: error && error.message ? error.message : "Upload authorization failed.",
      });
      writeJson(res, 400, {
        success: false,
        error: error.message || "Upload authorization failed.",
      });
    }
  };
}

module.exports = createBlobUploadHandler();
module.exports.createBlobUploadHandler = createBlobUploadHandler;
module.exports.validateUploadPathname = validateUploadPathname;
module.exports.parseClientPayload = parseClientPayload;
module.exports.inspectBlobPathname = inspectBlobPathname;
module.exports.ALLOWED_CONTENT_TYPES = ALLOWED_CONTENT_TYPES;
