import os
import uuid
import json
import math
import io
import base64
import re
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest, urlopen

import cv2
import numpy as np
from fastapi import Body, FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field
import runtime_paths as storage
from analysis_config import (
    COMPARISON_SIGNIFICANCE_THRESHOLD,
    HORSE_PROFILE_OPTIONS,
    DISCIPLINE_OPTIONS,
    confidence_band,
    evidence_status,
    get_discipline_config,
    get_profile_config,
    normalize_discipline,
    normalize_horse_profile,
    score_band,
    split_legacy_scheme,
)
from analysis_models import (
    AnalysisReportResponse,
    AnalysisResponse,
    ComparisonReportResponse,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonRow,
    ComparisonSide,
    MetricEntry,
    ScoreBlock,
    AnalysisRequest,
    VideoMetadata,
)

try:
    import mediapipe as mp  # type: ignore
    MP_AVAILABLE = hasattr(mp, "solutions")
except Exception:
    mp = None  # type: ignore
    MP_AVAILABLE = False

try:
    import multipart  # type: ignore
    MULTIPART_AVAILABLE = True
    MULTIPART_ERROR = ""
except Exception as exc:  # pragma: no cover - optional dependency
    MULTIPART_AVAILABLE = False
    MULTIPART_ERROR = str(exc)


_WEASYPRINT = None
_WEASYPRINT_ATTEMPTED = False

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov"}
ALLOWED_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime"}
BLOB_UPLOAD_PREFIX = "videos"
DEFAULT_VIDEO_UPLOAD_MAX_BYTES = 200 * 1024 * 1024


def get_video_upload_max_bytes() -> int:
    raw_value = os.getenv("VIDEO_UPLOAD_MAX_BYTES", str(DEFAULT_VIDEO_UPLOAD_MAX_BYTES))
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_VIDEO_UPLOAD_MAX_BYTES


def get_video_storage_access() -> str:
    access = os.getenv("VIDEO_STORAGE_ACCESS", "private").strip().lower()
    if access not in {"private", "public"}:
        return "private"
    return access


def sanitize_video_filename(filename: str, default_name: str = "uploaded_video.mp4") -> str:
    candidate = os.path.basename((filename or "").strip()) or default_name
    stem, ext = os.path.splitext(candidate)
    ext = ext.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        ext = ".mp4"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-") or "uploaded_video"
    return f"{safe_stem[:80]}{ext}"


def infer_video_content_type(filename: str, mime_type: Optional[str] = None) -> str:
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    if normalized in ALLOWED_VIDEO_MIME_TYPES:
        return normalized
    ext = os.path.splitext((filename or "").lower())[1]
    if ext == ".mov":
        return "video/quicktime"
    return "video/mp4"


def is_supported_video_filename(filename: str) -> bool:
    ext = os.path.splitext((filename or "").lower())[1]
    return ext in ALLOWED_VIDEO_EXTENSIONS


def build_blob_path(scope: str, original_filename: str, slot: str = "") -> str:
    safe_name = sanitize_video_filename(original_filename)
    parts = [BLOB_UPLOAD_PREFIX, scope]
    if slot:
        parts.append(slot)
    parts.append(f"{uuid.uuid4().hex}-{safe_name}")
    return "/".join(parts)


def validate_blob_path(pathname: str, scope: Optional[str] = None) -> str:
    normalized = (pathname or "").replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("Missing blob pathname.")
    parts = normalized.split("/")
    if parts[0] != BLOB_UPLOAD_PREFIX:
        raise ValueError("Invalid blob upload path.")
    if scope and len(parts) > 1 and parts[1] != scope:
        raise ValueError("Invalid blob upload scope.")
    basename = parts[-1]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", basename or ""):
        raise ValueError("Invalid blob filename.")
    ext = os.path.splitext(basename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("Only MP4 and MOV videos are allowed.")
    return normalized


def validate_video_reference_url(video_url: str) -> str:
    parsed = urlparse((video_url or "").strip())
    if parsed.scheme != "https":
        raise ValueError("Video storage URL must use HTTPS.")
    host = (parsed.netloc or "").lower()
    if not host.endswith(".blob.vercel-storage.com"):
        raise ValueError("Unsupported storage host.")
    return parsed.geturl()


async def write_uploadfile_to_path(upload: UploadFile, destination: Path, max_bytes: Optional[int] = None) -> int:
    max_bytes = max_bytes or get_video_upload_max_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with open(destination, "wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Video exceeds the configured maximum upload size.")
                handle.write(chunk)
    except Exception:
        if destination.exists():
            try:
                destination.unlink()
            except Exception:
                pass
        raise
    finally:
        try:
            await upload.close()
        except Exception:
            pass
    return total


def download_blob_to_path(video_url: str, destination: Path, max_bytes: Optional[int] = None) -> int:
    validated_url = validate_video_reference_url(video_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = max_bytes or get_video_upload_max_bytes()
    parsed = urlparse(validated_url)
    headers = {}
    if ".private.blob.vercel-storage.com" in parsed.netloc.lower():
        token = os.getenv("BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_OIDC_TOKEN")
        if not token:
            raise ValueError("Missing blob download token.")
        headers["Authorization"] = f"Bearer {token}"
    request = UrlRequest(validated_url, headers=headers)
    total = 0
    try:
        with urlopen(request, timeout=120) as response, open(destination, "wb") as handle:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    expected_size = int(content_length)
                except (TypeError, ValueError):
                    expected_size = None
                else:
                    if expected_size > max_bytes:
                        raise ValueError("Video exceeds the configured maximum upload size.")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Video exceeds the configured maximum upload size.")
                handle.write(chunk)
    except Exception:
        if destination.exists():
            try:
                destination.unlink()
            except Exception:
                pass
        raise
    return total


def get_weasyprint():
    global _WEASYPRINT_ATTEMPTED, _WEASYPRINT
    if _WEASYPRINT_ATTEMPTED:
        return _WEASYPRINT
    _WEASYPRINT_ATTEMPTED = True
    try:
        import weasyprint as weasyprint_module  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        _WEASYPRINT = None
    else:
        _WEASYPRINT = weasyprint_module
    return _WEASYPRINT


def clamp_point(x: float, y: float, w: int, h: int) -> List[float]:
    return [float(max(0, min(w - 1, x))), float(max(0, min(h - 1, y)))]

def detect_rider_pose_points(frame_bgr: np.ndarray):
    """
    Returns keypoints and a visibility score.
    """
    if not MP_AVAILABLE:
        return None
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose  # type: ignore[attr-defined]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
    ) as pose:
        res = pose.process(frame_rgb)

    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    pose_lm = mp_pose.PoseLandmark

    lhip = lm[pose_lm.LEFT_HIP]
    rhip = lm[pose_lm.RIGHT_HIP]
    lsh = lm[pose_lm.LEFT_SHOULDER]
    rsh = lm[pose_lm.RIGHT_SHOULDER]

    hip_vis = (lhip.visibility + rhip.visibility) / 2.0
    sh_vis = (lsh.visibility + rsh.visibility) / 2.0
    vis = float(min(hip_vis, sh_vis))

    def to_xy(pt):
        return (float(pt.x * w), float(pt.y * h))

    return {
        "left_hip": to_xy(lhip),
        "right_hip": to_xy(rhip),
        "left_shoulder": to_xy(lsh),
        "right_shoulder": to_xy(rsh),
        "hip_mid": ((lhip.x + rhip.x) * 0.5 * w, (lhip.y + rhip.y) * 0.5 * h),
        "sh_mid": ((lsh.x + rsh.x) * 0.5 * w, (lsh.y + rsh.y) * 0.5 * h),
        "visibility": vis,
    }



def detect_horse_topline_points(frame_bgr: np.ndarray):
    """
    Returns (withers_xy, croup_xy, contour_score)
    contour_score ~ confidence based on contour size
    """
    h, w = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge-based silhouette (works decently for side-view)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    # Largest contour assumed horse+rider mass
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    area_score = min(1.0, area / (w * h * 0.20))  # 20% of frame as "good"

    if area < (w * h * 0.05):  # too small
        return None, None, area_score

    x, y, bw, bh = cv2.boundingRect(cnt)

    # Build a topline by scanning x columns: pick min y (highest pixel) inside contour.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    xs = list(range(x, x + bw))
    top_points = []
    for xi in xs:
        col = mask[:, xi]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            top_points.append((xi, int(ys.min())))

    if len(top_points) < 30:
        return None, None, area_score

    # Split topline into thirds
    n = len(top_points)
    front = top_points[: n // 3]
    back  = top_points[2 * n // 3 :]

    # "Highest" = minimum y
    withers = min(front, key=lambda p: p[1])
    croup   = min(back,  key=lambda p: p[1])

    return (float(withers[0]), float(withers[1])), (float(croup[0]), float(croup[1])), area_score


def safe_crop(frame_bgr: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> Optional[np.ndarray]:
    """
    Crop helper that clamps bounds to the frame and returns None if invalid.
    """
    h, w = frame_bgr.shape[:2]
    xi0 = int(max(0, min(w - 1, x0)))
    xi1 = int(max(0, min(w, x1)))
    yi0 = int(max(0, min(h - 1, y0)))
    yi1 = int(max(0, min(h, y1)))
    if xi1 <= xi0 or yi1 <= yi0:
        return None
    return frame_bgr[yi0:yi1, xi0:xi1]


def get_pose_landmarks(frame_bgr: np.ndarray):
    """
    Run MediaPipe pose once to reuse landmarks for gear detection heuristics.
    """
    if not MP_AVAILABLE:
        return None
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(frame_rgb)
    if not res.pose_landmarks:
        return None
    return {"landmarks": res.pose_landmarks.landmark, "pose_lm": mp_pose.PoseLandmark, "w": w, "h": h}


def compute_dark_edge_score(patch: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """
    Lightweight texture score: combines darkness and edge density to guess solid gear pieces.
    """
    if patch is None or patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32, copy=False)
    dark_ratio = float(np.mean(gray_f < 90.0))
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.mean(edges > 0))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1].astype(np.float32, copy=False)) / 255.0)
    return {
        "dark_ratio": dark_ratio,
        "edge_density": edge_density,
        "sat_mean": sat_mean,
        "score": float(0.5 * dark_ratio + 0.35 * edge_density + 0.15 * (1.0 - sat_mean)),
    }


def compute_knee_angle(hip_xy: Tuple[float, float], knee_xy: Tuple[float, float], ankle_xy: Tuple[float, float]) -> float:
    """
    Returns the knee angle in degrees using three pose points.
    """
    v1 = np.array([hip_xy[0] - knee_xy[0], hip_xy[1] - knee_xy[1]])
    v2 = np.array([ankle_xy[0] - knee_xy[0], ankle_xy[1] - knee_xy[1]])
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return 0.0
    cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def auto_detect_gear_and_safety(frame_bgr: np.ndarray, points: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
    """
    Heuristic gear detection using the first frame and pose landmarks.
    Returns gear guesses, per-item confidence, and notes for the report.
    """
    detected = {
        "helmet": "unknown",
        "boots": "unknown",
        "pad": "unknown",
        "girth": "unknown",
        "breastplate": "unknown",
        "crupper": "unknown",
        "stirrups": "unknown",
    }
    confidences: Dict[str, float] = {k: 0.0 for k in detected}
    notes: List[str] = []

    pose_data = get_pose_landmarks(frame_bgr)
    if pose_data is None:
        notes.append("Pose landmarks unavailable; gear auto-detect limited.")
        return {"gear": detected, "confidences": confidences, "notes": notes, "method": "frame_only"}

    lms = pose_data["landmarks"]
    pose_lm = pose_data["pose_lm"]
    w = pose_data["w"]
    h = pose_data["h"]

    # Helmet detection: dark, low-saturation blob above face landmarks.
    head_pts = []
    for idx in [pose_lm.NOSE, pose_lm.LEFT_EAR, pose_lm.RIGHT_EAR, pose_lm.LEFT_EYE, pose_lm.RIGHT_EYE]:
        lm = lms[idx]
        if lm.visibility > 0.25:
            head_pts.append((lm.x * w, lm.y * h))
    if len(head_pts) >= 2:
        xs = [p[0] for p in head_pts]
        ys = [p[1] for p in head_pts]
        x0 = min(xs) - 0.08 * w
        x1 = max(xs) + 0.08 * w
        y_mid = min(ys)
        y0 = y_mid - 0.22 * h
        y1 = y_mid + 0.12 * h
        patch = safe_crop(frame_bgr, x0, y0, x1, y1)
        stats = compute_dark_edge_score(patch)
        if stats:
            score = stats["score"]
            confidences["helmet"] = float(max(0.0, min(1.0, score)))
            if score > 0.42:
                detected["helmet"] = "yes"
            elif score < 0.24:
                detected["helmet"] = "no"
            notes.append(f"Helmet score {score:.2f} (dark ratio {stats['dark_ratio']:.2f}).")
    else:
        notes.append("Head landmarks too low confidence for helmet check.")

    # Boots detection: dark, high-texture region around ankles.
    boot_scores = []
    for knee_id, ankle_id in [(pose_lm.LEFT_KNEE, pose_lm.LEFT_ANKLE), (pose_lm.RIGHT_KNEE, pose_lm.RIGHT_ANKLE)]:
        knee = lms[knee_id]
        ankle = lms[ankle_id]
        if knee.visibility < 0.20 or ankle.visibility < 0.20:
            continue
        x0 = min(knee.x, ankle.x) * w - 0.04 * w
        x1 = max(knee.x, ankle.x) * w + 0.04 * w
        y0 = min(knee.y, ankle.y) * h - 0.02 * h
        y1 = max(knee.y, ankle.y) * h + 0.10 * h
        patch = safe_crop(frame_bgr, x0, y0, x1, y1)
        stats = compute_dark_edge_score(patch)
        if stats:
            boot_scores.append(stats["score"])
    if boot_scores:
        avg_score = float(np.mean(boot_scores))
        confidences["boots"] = float(min(1.0, max(boot_scores)))
        if avg_score > 0.36:
            detected["boots"] = "yes"
        elif avg_score < 0.20:
            detected["boots"] = "no"
        notes.append(f"Boots score {avg_score:.2f} from {len(boot_scores)} leg(s).")
    else:
        notes.append("Boot region not clear; could not auto-check boots.")

    # Stirrups length inference from knee angle.
    angles = []
    for hip_id, knee_id, ankle_id in [
        (pose_lm.LEFT_HIP, pose_lm.LEFT_KNEE, pose_lm.LEFT_ANKLE),
        (pose_lm.RIGHT_HIP, pose_lm.RIGHT_KNEE, pose_lm.RIGHT_ANKLE),
    ]:
        hip = lms[hip_id]
        knee = lms[knee_id]
        ankle = lms[ankle_id]
        if hip.visibility < 0.20 or knee.visibility < 0.20 or ankle.visibility < 0.20:
            continue
        angle = compute_knee_angle((hip.x * w, hip.y * h), (knee.x * w, knee.y * h), (ankle.x * w, ankle.y * h))
        if angle > 0:
            angles.append(angle)
    if angles:
        ang = float(np.mean(angles))
        confidences["stirrups"] = float(0.4 + 0.3 * min(1.0, len(angles) / 2))
        if ang < 130:
            detected["stirrups"] = "short"
        elif ang > 165:
            detected["stirrups"] = "long"
        else:
            detected["stirrups"] = "medium"
        notes.append(f"Stirrup estimate {detected['stirrups']} (knee angle ~{ang:.1f} deg).")

    # Pad detection: use saddle band edges if points are known.
    if points and points.get("saddle_front") and points.get("saddle_rear"):
        sf = points["saddle_front"]
        sr = points["saddle_rear"]
        x0 = min(sf[0], sr[0]) - 0.05 * w
        x1 = max(sf[0], sr[0]) + 0.05 * w
        y_center = (sf[1] + sr[1]) * 0.5
        y0 = y_center - 0.08 * h
        y1 = y_center + 0.14 * h
        patch = safe_crop(frame_bgr, x0, y0, x1, y1)
        if patch is not None:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray_f = gray.astype(np.float32, copy=False)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 25, 80)
            edge_density = float(np.mean(edges > 0))
            contrast = float(np.std(gray_f) / 255.0)
            pad_score = float(0.6 * edge_density + 0.4 * contrast)
            confidences["pad"] = float(min(1.0, pad_score * 2.0))
            if pad_score > 0.22:
                detected["pad"] = "halfpad"
            else:
                detected["pad"] = "normal"
            notes.append(f"Pad texture score {pad_score:.2f} (contrast {contrast:.2f}).")

    return {"gear": detected, "confidences": confidences, "notes": notes, "method": "pose_first_frame"}


def merge_gear_sources(primary: Dict[str, str], fallback: Dict[str, str]) -> Dict[str, str]:
    """
    Use primary values when present, otherwise fallback. Unknown/empty entries are skipped.
    """
    merged = dict(fallback or {})
    for key, val in (primary or {}).items():
        if val and val != "unknown":
            merged[key] = val
    return merged


def auto_calibrate_points(
    frame_bgr: np.ndarray,
    horse_profile: str,
    saddle_type: str,
    discipline: str = "general_riding",
) -> Dict:
    """
    Returns:
      {
        "points": {saddle_front:[x,y], saddle_rear:[x,y], withers:[x,y], croup:[x,y]},
        "confidence": float,
        "details": {...}
      }
    """
    h, w = frame_bgr.shape[:2]
    profile_key = normalize_horse_profile(horse_profile)
    discipline_key = normalize_discipline(discipline)
    profile_cfg = get_profile_config(profile_key)
    discipline_cfg = get_discipline_config(discipline_key)

    pose = detect_rider_pose_points(frame_bgr)
    withers, croup, contour_score = detect_horse_topline_points(frame_bgr)

    if withers is None or croup is None:
        # Cannot auto-detect topline
        return {"points": None, "confidence": 0.0, "details": {"pose_vis": 0.0, "contour_score": contour_score}}

    # Estimate saddle center under pelvis (hip_mid)
    if pose is not None:
        saddle_center_x, saddle_center_y = pose["hip_mid"]
    else:
        saddle_center_x = (withers[0] + croup[0]) / 2.0
        saddle_center_y = (withers[1] + croup[1]) / 2.0 + 0.05 * h

    # Estimate saddle direction roughly horizontal
    # Use shoulder->hip vector to adapt to rider posture slightly
    dx = 1.0
    if pose is not None and pose["sh_mid"] is not None:
        dx = (saddle_center_x - pose["sh_mid"][0])
        if abs(dx) < 1e-6:
            dx = 1.0
    dir_sign = 1.0 if dx >= 0 else -1.0

    # Offsets: tune by discipline / horse profile / saddle type
    # (These are MVP defaults; we will tune later using sample videos.)
    base_len = 0.10 * w  # 10% of frame width
    if saddle_type == "western":
        base_len = 0.12 * w
    if profile_key in ["round_barrel", "wide_build"]:
        base_len *= 1.03
    if discipline_key in ["polo", "eventing", "barrel_racing", "racing_gallop"]:
        base_len *= 1.05
    if profile_cfg.get("clearance_threshold_adjust", 0.0):
        base_len += float(profile_cfg.get("clearance_threshold_adjust", 0.0)) * 0.04
    if discipline_cfg.get("expected_motion") in {"dynamic", "forward"}:
        base_len *= 1.02

    saddle_front = (saddle_center_x + dir_sign * base_len * 0.5, saddle_center_y)
    saddle_rear  = (saddle_center_x - dir_sign * base_len * 0.5, saddle_center_y)

    pose = detect_rider_pose_points(frame_bgr)
    withers, croup, contour_score = detect_horse_topline_points(frame_bgr)

    if withers is None or croup is None:
        return {"points": None, "confidence": 0.0, "details": {"reason": "topline_not_found"}}

    pose_vis = pose["visibility"] if pose is not None else 0.0
    confidence = float(0.6 * pose_vis + 0.4 * contour_score)

    points = {
        "saddle_front": clamp_point(saddle_front[0], saddle_front[1], w, h),
        "saddle_rear": clamp_point(saddle_rear[0], saddle_rear[1], w, h),
        "withers": clamp_point(withers[0], withers[1], w, h),
        "croup": clamp_point(croup[0], croup[1], w, h),
        "left_shoulder": clamp_point(
            pose["left_shoulder"][0] if pose is not None else withers[0] - 0.05 * w,
            pose["left_shoulder"][1] if pose is not None else withers[1] + 10,
            w,
            h,
        ),
        "right_shoulder": clamp_point(
            pose["right_shoulder"][0] if pose is not None else withers[0] + 0.05 * w,
            pose["right_shoulder"][1] if pose is not None else withers[1] + 10,
            w,
            h,
        ),
        "left_hip": clamp_point(
            pose["left_hip"][0] if pose is not None else croup[0] - 0.05 * w,
            pose["left_hip"][1] if pose is not None else croup[1] + 10,
            w,
            h,
        ),
        "right_hip": clamp_point(
            pose["right_hip"][0] if pose is not None else croup[0] + 0.05 * w,
            pose["right_hip"][1] if pose is not None else croup[1] + 10,
            w,
            h,
        ),
    }

    return {"points": points, "confidence": confidence, "details": {"pose_vis": pose_vis, "contour_score": contour_score}}

app = FastAPI(docs_url="/docs", redoc_url="/redoc")

BASE_DIR = storage.PROJECT_ROOT
RUNTIME_ROOT = storage.RUNTIME_ROOT
UPLOAD_DIR = storage.UPLOAD_DIR
OUTPUT_DIR = storage.OUTPUT_DIR
OUTPUTS_DIR = OUTPUT_DIR
LEGACY_OUTPUTS_DIR = BASE_DIR / "outputs"
COMPARE_DIR = storage.COMPARE_DIR
REPORT_DIR = storage.REPORT_DIR
TEMP_DIR = storage.TEMP_DIR
IS_VERCEL = storage.IS_VERCEL

# --------- Horse profile / discipline selectors ----------
HORSE_SCHEMES = HORSE_PROFILE_OPTIONS


def horse_profile_select_html(selected: str = "high_wither") -> str:
    options: List[str] = []
    for key, label in HORSE_PROFILE_OPTIONS:
        sel = "selected" if key == selected else ""
        options.append(f'<option value="{key}" {sel}>{label}</option>')
    return "\n".join(options)


def discipline_select_html(selected: str = "general_riding") -> str:
    options: List[str] = []
    for key, label in DISCIPLINE_OPTIONS:
        sel = "selected" if key == selected else ""
        options.append(f'<option value="{key}" {sel}>{label}</option>')
    return "\n".join(options)


@app.get("/horse-scheme-guide", response_class=HTMLResponse)
def horse_scheme_guide():
    guide_path = os.path.join(BASE_DIR, "horse_scheme_guide.html")
    if not os.path.exists(guide_path):
        return HTMLResponse("<h3>Guide not found</h3>", status_code=404)
    return open(guide_path, "r", encoding="utf-8").read()


@app.get("/discipline-guide", response_class=HTMLResponse)
def discipline_guide():
    guide_path = os.path.join(BASE_DIR, "discipline_guide.html")
    if not os.path.exists(guide_path):
        return HTMLResponse("<h3>Guide not found</h3>", status_code=404)
    return open(guide_path, "r", encoding="utf-8").read()


@app.get("/point-selection-guide", response_class=HTMLResponse)
def point_selection_guide():
    guide_path = os.path.join(BASE_DIR, "point_selection_guide.html")
    if not os.path.exists(guide_path):
        return HTMLResponse("<h3>Guide not found</h3>", status_code=404)
    return open(guide_path, "r", encoding="utf-8").read()


def schemes_select_html(selected: str = "high_wither") -> str:
    options: List[str] = []
    for key, label in HORSE_SCHEMES:
        sel = "selected" if key == selected else ""
        options.append(f'<option value="{key}" {sel}>{label}</option>')
    return "\n".join(options)


def normalize_points(points: dict) -> Dict[str, List[float]]:
    """
    Convert JSON points to strict numeric [x,y] floats:
    {"name": [x, y]}
    """
    out: Dict[str, List[float]] = {}
    for k, v in points.items():
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError(f"Invalid point format for {k}. Expected [x,y].")
        out[k] = [float(v[0]), float(v[1])]
    return out


def first_existing_path(*paths: os.PathLike[str] | str) -> Optional[Path]:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


# ------------------- Utility: extract first frame -------------------
def extract_first_frame(video_path: str, out_png_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(os.fspath(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame from video.")
    h, w = frame.shape[:2]
    cv2.imwrite(os.fspath(out_png_path), frame)
    return w, h


# ------------------- Utility: extract frames at target FPS -------------------
def extract_frames(video_path: str, target_fps: int = 12) -> Tuple[List[np.ndarray], List[float]]:
    cap = cv2.VideoCapture(os.fspath(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(native_fps // target_fps), 1)

    frames: List[np.ndarray] = []
    times: List[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append(frame)
            times.append(float(t))
        i += 1
    cap.release()

    if len(frames) < 5:
        raise RuntimeError("Video too short or cannot extract frames.")
    return frames, times


# ------------------- Tracking: Lucas-Kanade optical flow -------------------
def track_points_lk(
    frames: List[np.ndarray],
    init_points: Dict[str, List[float]]
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, Any]]:
    names = list(init_points.keys())
    validated_points: Dict[str, Tuple[float, float]] = {}
    for n in names:
        coords = init_points[n]
        if len(coords) != 2:
            raise ValueError(f"Point '{n}' must have exactly two coordinates.")
        validated_points[n] = (float(coords[0]), float(coords[1]))

    p0 = np.array([validated_points[n] for n in names], dtype=np.float32).reshape(-1, 1, 2)

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    tracks: Dict[str, List[Tuple[float, float]]] = {n: [validated_points[n]] for n in names}
    success_count = 0
    total_count = 0
    reinit_count = 0

    for idx in range(1, len(frames)):
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)

        p1_init = p0.copy()  # avoid passing None (Pylance-friendly)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            p0,
            p1_init,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if p1 is None or st is None:
            for n in names:
                tracks[n].append(tracks[n][-1])
            reinit_count += 1
        else:
            st_flat = st.reshape(-1)
            p1r = p1.reshape(-1, 2)
            for j, n in enumerate(names):
                if int(st_flat[j]) == 1:
                    tracks[n].append((float(p1r[j][0]), float(p1r[j][1])))
                    success_count += 1
                else:
                    tracks[n].append(tracks[n][-1])
                total_count += 1

            p0 = p1
            prev_gray = gray

    total_count = max(total_count, 1)
    stats = {
        "tracking_success_pct": float((success_count / total_count) * 100.0),
        "frames": len(frames),
        "reinitializations": reinit_count,
        "frame_w": frames[0].shape[1] if frames else 0,
        "frame_h": frames[0].shape[0] if frames else 0,
    }

    return tracks, stats


# ------------------- Metrics -------------------
def angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def compute_metrics(
    tracks: Dict[str, List[Tuple[float, float]]],
    times: List[float],
    tracking_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if len(times) == 0:
        raise ValueError("No frames available to compute metrics.")
    tracking_stats = tracking_stats or {}

    sf = tracks["saddle_front"]
    sr = tracks["saddle_rear"]
    wh = tracks["withers"]
    cr = tracks["croup"]
    ls = tracks["left_shoulder"]
    rs = tracks["right_shoulder"]
    lh = tracks["left_hip"]
    rh = tracks["right_hip"]

    pitch = [angle_deg(sf[i], sr[i]) for i in range(len(times))]
    backline = [angle_deg(wh[i], cr[i]) for i in range(len(times))]
    shoulder_span = [rs[i][0] - ls[i][0] for i in range(len(times))]
    hip_span = [rh[i][0] - lh[i][0] for i in range(len(times))]
    shoulder_height_diff = [rs[i][1] - ls[i][1] for i in range(len(times))]
    hip_height_diff = [rh[i][1] - lh[i][1] for i in range(len(times))]

    mid = [((sf[i][0] + sr[i][0]) / 2.0, (sf[i][1] + sr[i][1]) / 2.0) for i in range(len(times))]
    mid_x = [m[0] for m in mid]
    mid_y = [m[1] for m in mid]

    p5, p95 = np.percentile(pitch, 5), np.percentile(pitch, 95)
    rock_amp = float(p95 - p5)
    pitch_std = float(np.std(pitch))

    cadence = 0.0
    if len(times) > 4:
        mid_y_centered = np.array(mid_y) - np.mean(mid_y)
        zero_cross = np.where(np.diff(np.sign(mid_y_centered)))[0]
        if len(zero_cross) > 1:
            duration = times[zero_cross[-1]] - times[zero_cross[0]]
            cycles = len(zero_cross) / 2.0
            if duration > 0:
                cadence = float(cycles / duration)

    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    mid_y_range = float(max(mid_y) - min(mid_y))
    drift_rate = float((mid_x[-1] - mid_x[0]) / duration) if duration > 0 else 0.0
    drift_direction = "forward" if drift_rate > 1.0 else "backward" if drift_rate < -1.0 else "stable"

    pitch_backline_diff = float(np.mean(pitch) - np.mean(backline))

    # Clearance proxies
    front_clear = [math.dist(sf[i], wh[i]) for i in range(len(times))]
    rear_clear = [math.dist(sr[i], cr[i]) for i in range(len(times))]
    clearance_min = float(min(front_clear)) if front_clear else 0.0
    clearance_mean = float(np.mean(front_clear)) if front_clear else 0.0
    clearance_std = float(np.std(front_clear)) if front_clear else 0.0
    rear_clear_std = float(np.std(rear_clear)) if rear_clear else 0.0

    frame_h = float(tracking_stats.get("frame_h", 0) or 0)
    clearance_threshold = max(10.0, frame_h * 0.025) if frame_h else 12.0
    clearance_collapse = clearance_min < clearance_threshold
    bridging_proxy = rear_clear_std - clearance_std

    # Symmetry
    shoulder_diff = [rs[i][1] - ls[i][1] for i in range(len(times))]
    hip_diff = [rh[i][1] - lh[i][1] for i in range(len(times))]
    shoulder_angle = [angle_deg(ls[i], rs[i]) for i in range(len(times))]

    rider_sym = {
        "shoulder_diff_mean": float(np.mean(shoulder_diff)),
        "shoulder_diff_std": float(np.std(shoulder_diff)),
        "hip_diff_mean": float(np.mean(hip_diff)),
        "hip_diff_std": float(np.std(hip_diff)),
        "lean_consistency_std": float(np.std(shoulder_angle)),
        "confidence": tracking_stats.get("tracking_success_pct", 0.0),
    }

    saddle_sym = {
        "roll_proxy": float(np.std(shoulder_diff) + np.std(hip_diff)),
        "confidence": tracking_stats.get("tracking_success_pct", 0.0),
        "note": "Limited: side-view roll estimation is approximate.",
    }

    horse_sym = {
        "topline_std": float(np.std(backline)),
        "confidence": tracking_stats.get("tracking_success_pct", 0.0),
        "note": "Side-view only; lateral symmetry not observed.",
    }

    tracking_conf = {
        "tracking_success_pct": float(tracking_stats.get("tracking_success_pct", 0.0)),
        "frames": float(tracking_stats.get("frames", len(times))),
        "reinitializations": float(tracking_stats.get("reinitializations", 0)),
    }

    # Downsample series for lightweight charting
    def downsample(series: List[float], max_points: int = 120) -> List[float]:
        if len(series) <= max_points:
            return [float(x) for x in series]
        step = max(1, len(series) // max_points)
        return [float(series[i]) for i in range(0, len(series), step)]

    series = {
        "time": downsample(times),
        "pitch": downsample(pitch),
        "drift_x": downsample(mid_x),
        "drift_y": downsample(mid_y),
    }

    return {
        "pitch_mean_deg": float(np.mean(pitch)),
        "pitch_std_deg": pitch_std,
        "rock_amplitude_deg": rock_amp,
        "mid_drift_x_px": float(max(mid_x) - min(mid_x)),
        "mid_drift_y_px": float(max(mid_y) - min(mid_y)),
        "mid_drift_direction": drift_direction,
        "mid_drift_rate_px_s": drift_rate,
        "mid_bounce_y_px": mid_y_range,
        "backline_mean_deg": float(np.mean(backline)),
        "backline_std_deg": float(np.std(backline)),
        "shoulder_level_std_px": float(np.std(shoulder_height_diff)),
        "hip_level_std_px": float(np.std(hip_height_diff)),
        "shoulder_span_mean_px": float(np.mean(shoulder_span)),
        "hip_span_mean_px": float(np.mean(hip_span)),
        "cadence_hz": cadence,
        "frames_analyzed": float(len(times)),
        "duration_sec": duration,
        "series": series,
        "alignment": {
          "topline_mean_deg": float(np.mean(backline)),
          "topline_std_deg": float(np.std(backline)),
          "saddle_topline_diff_deg": pitch_backline_diff,
          "withers_clearance_min_px": clearance_min,
          "withers_clearance_mean_px": clearance_mean,
          "withers_clearance_std_px": clearance_std,
          "clearance_collapse": clearance_collapse,
          "bridging_proxy": bridging_proxy,
        },
        "symmetry": {
          "rider": rider_sym,
          "saddle": saddle_sym,
          "horse": horse_sym,
        },
        "tracking": tracking_conf,
    }


def model_to_dict(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[attr-defined]
    if hasattr(model, "dict"):
        return model.dict()  # type: ignore[call-arg]
    if isinstance(model, dict):
        return dict(model)
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return float(default)
    return float(np.mean(arr))


def _safe_std(values: List[float], default: float = 0.0) -> float:
    arr = [float(v) for v in values if v is not None]
    if len(arr) < 2:
        return float(default)
    return float(np.std(arr))


def _score_value(value: float, ceiling: float = 100.0) -> int:
    return int(max(0.0, min(ceiling, round(value))))


def _score_from_distance(value: Optional[float], target: float, spread: float, weight: float = 1.0) -> float:
    if value is None:
        return 0.0
    spread = max(spread, 1e-6)
    delta = abs(float(value) - float(target))
    return max(0.0, 100.0 - (delta / spread) * 100.0 * weight)


def _score_from_std(std_value: Optional[float], spread: float, base: float = 100.0, weight: float = 1.0) -> float:
    if std_value is None:
        return 0.0
    spread = max(spread, 1e-6)
    return max(0.0, base - (float(std_value) / spread) * 100.0 * weight)


def resolve_analysis_context(meta: Dict[str, Any]) -> Tuple[str, str, str]:
    legacy_profile, legacy_discipline = split_legacy_scheme(meta.get("horse_scheme"))
    horse_profile = normalize_horse_profile(meta.get("horse_profile") or legacy_profile or meta.get("horse_scheme"))
    discipline = normalize_discipline(meta.get("discipline") or legacy_discipline)
    saddle_type = (meta.get("saddle_type") or "english").strip().lower()
    if saddle_type not in {"english", "western"}:
        saddle_type = "english"
    return horse_profile, discipline, saddle_type


def build_video_metadata(meta: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> VideoMetadata:
    metrics = metrics or {}
    fps = None
    frames = None
    if metrics:
        tracking = metrics.get("tracking", {}) or {}
        frames = int(tracking.get("frames", metrics.get("frames_analyzed", 0)) or 0)
        duration = float(metrics.get("duration_sec", 0.0) or 0.0)
        if duration > 0 and frames:
            fps = float(frames / duration)
    display_name = str(meta.get("original_filename") or meta.get("video_filename", ""))
    video_name = str(meta.get("video_filename", ""))
    return VideoMetadata(
        filename=display_name or video_name,
        mime_type=guess_mime(video_name) if video_name else "",
        duration_sec=float(metrics.get("duration_sec", 0.0) or 0.0) or None,
        width=int(meta.get("frame_width", 0) or 0) or None,
        height=int(meta.get("frame_height", 0) or 0) or None,
        fps=fps,
        frames=frames,
    )


def make_metric_entry(
    name: str,
    value: Optional[float],
    unit: str = "",
    status: str = "Insufficient Data",
    source: str = "estimated",
    note: str = "",
    precision: int = 1,
) -> MetricEntry:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MetricEntry(
            name=name,
            value=None,
            display="N/A",
            unit=unit,
            status="Insufficient Data",
            source="insufficient",
            note=note,
        )
    if precision <= 0:
        display = f"{int(round(float(value)))}"
    else:
        display = f"{float(value):.{precision}f}"
    return MetricEntry(
        name=name,
        value=float(value),
        display=display,
        unit=unit,
        status=status,
        source=source,
        note=note,
    )


def encode_frame_data_uri(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def draw_annotation_overlay(
    frame_bgr: np.ndarray,
    points: Optional[Dict[str, List[float]]] = None,
    pose_sample: Optional[Dict[str, Any]] = None,
    title: str = "",
) -> np.ndarray:
    annotated = frame_bgr.copy()
    overlay = annotated

    if points:
        colors = {
            "saddle_front": (45, 212, 191),
            "saddle_rear": (96, 165, 250),
            "withers": (34, 197, 94),
            "croup": (249, 115, 22),
            "left_shoulder": (168, 85, 247),
            "right_shoulder": (168, 85, 247),
            "left_hip": (236, 72, 153),
            "right_hip": (236, 72, 153),
        }
        for key, point in points.items():
            if not point or len(point) != 2:
                continue
            px, py = int(round(point[0])), int(round(point[1]))
            color = colors.get(key, (255, 255, 255))
            cv2.circle(overlay, (px, py), 5, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                key.replace("_", " "),
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                lineType=cv2.LINE_AA,
            )
        if points.get("saddle_front") and points.get("saddle_rear"):
            sf = points["saddle_front"]
            sr = points["saddle_rear"]
            cv2.line(overlay, (int(sf[0]), int(sf[1])), (int(sr[0]), int(sr[1])), (45, 212, 191), 2, lineType=cv2.LINE_AA)
        if points.get("withers") and points.get("croup"):
            wh = points["withers"]
            cr = points["croup"]
            cv2.line(overlay, (int(wh[0]), int(wh[1])), (int(cr[0]), int(cr[1])), (34, 197, 94), 2, lineType=cv2.LINE_AA)

    if pose_sample:
        joints = pose_sample.get("joints", {})
        joint_colors = {
            "left_shoulder": (255, 255, 255),
            "right_shoulder": (255, 255, 255),
            "left_hip": (255, 255, 255),
            "right_hip": (255, 255, 255),
            "left_knee": (255, 230, 109),
            "right_knee": (255, 230, 109),
            "left_ankle": (255, 230, 109),
            "right_ankle": (255, 230, 109),
            "nose": (251, 191, 36),
        }
        for key, point in joints.items():
            if not point or len(point) != 2:
                continue
            px, py = int(round(point[0])), int(round(point[1]))
            cv2.circle(overlay, (px, py), 4, joint_colors.get(key, (255, 255, 255)), -1, lineType=cv2.LINE_AA)
        for pair in [
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
        ]:
            a = joints.get(pair[0])
            b = joints.get(pair[1])
            if a and b:
                cv2.line(
                    overlay,
                    (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))),
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )
        if pose_sample.get("torso_mid") and pose_sample.get("hip_mid"):
            torso_mid = pose_sample["torso_mid"]
            hip_mid = pose_sample["hip_mid"]
            cv2.line(
                overlay,
                (int(round(hip_mid[0])), int(round(hip_mid[1]))),
                (int(round(torso_mid[0])), int(round(torso_mid[1]))),
                (56, 189, 248),
                2,
                lineType=cv2.LINE_AA,
            )

    if title:
        cv2.rectangle(overlay, (10, 10), (min(470, overlay.shape[1] - 10), 62), (15, 23, 42), -1)
        cv2.putText(
            overlay,
            title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return annotated


def sample_rider_pose_metrics(
    frames: List[np.ndarray],
    times: List[float],
    points: Optional[Dict[str, List[float]]] = None,
    max_samples: int = 8,
) -> Dict[str, Any]:
    if not frames or not times:
        return {
            "available": False,
            "estimated": True,
            "sample_count": 0,
            "confidence_pct": 0.0,
            "samples": [],
            "series": {},
            "summary": {},
            "notes": ["No frames available for pose sampling."],
            "best_frame_index": None,
            "annotated_frame": "",
        }

    sample_total = max(1, min(int(max_samples), len(frames)))
    sample_indices = sorted(set(np.linspace(0, len(frames) - 1, sample_total).astype(int).tolist()))
    samples: List[Dict[str, Any]] = []
    series: Dict[str, List[float]] = {
        "time": [],
        "torso_angle": [],
        "head_offset": [],
        "shoulder_level": [],
        "hip_level": [],
        "seat_offset": [],
        "left_knee_angle": [],
        "right_knee_angle": [],
        "left_ankle_angle": [],
        "right_ankle_angle": [],
        "vertical_motion": [],
        "horizontal_motion": [],
    }
    visibilities: List[float] = []

    saddle_mid = None
    if points and points.get("saddle_front") and points.get("saddle_rear"):
        saddle_mid = (
            float((points["saddle_front"][0] + points["saddle_rear"][0]) / 2.0),
            float((points["saddle_front"][1] + points["saddle_rear"][1]) / 2.0),
        )

    for idx in sample_indices:
        frame = frames[idx]
        pose_data = get_pose_landmarks(frame)
        if pose_data is None:
            continue

        lms = pose_data["landmarks"]
        pose_lm = pose_data["pose_lm"]
        w = float(pose_data["w"])
        h = float(pose_data["h"])

        def xy(pt) -> Tuple[float, float]:
            return (float(pt.x * w), float(pt.y * h))

        def maybe_xy(pt):
            if pt.visibility < 0.15:
                return None
            return xy(pt)

        nose = maybe_xy(lms[pose_lm.NOSE])
        lsh = maybe_xy(lms[pose_lm.LEFT_SHOULDER])
        rsh = maybe_xy(lms[pose_lm.RIGHT_SHOULDER])
        lhip = maybe_xy(lms[pose_lm.LEFT_HIP])
        rhip = maybe_xy(lms[pose_lm.RIGHT_HIP])
        lknee = maybe_xy(lms[pose_lm.LEFT_KNEE])
        rknee = maybe_xy(lms[pose_lm.RIGHT_KNEE])
        lankle = maybe_xy(lms[pose_lm.LEFT_ANKLE])
        rankle = maybe_xy(lms[pose_lm.RIGHT_ANKLE])
        lheel = maybe_xy(lms[pose_lm.LEFT_HEEL]) if hasattr(pose_lm, "LEFT_HEEL") else None
        rheel = maybe_xy(lms[pose_lm.RIGHT_HEEL]) if hasattr(pose_lm, "RIGHT_HEEL") else None
        lfoot = maybe_xy(lms[pose_lm.LEFT_FOOT_INDEX]) if hasattr(pose_lm, "LEFT_FOOT_INDEX") else None
        rfoot = maybe_xy(lms[pose_lm.RIGHT_FOOT_INDEX]) if hasattr(pose_lm, "RIGHT_FOOT_INDEX") else None

        coords = [p for p in [nose, lsh, rsh, lhip, rhip, lknee, rknee, lankle, rankle] if p is not None]
        if len(coords) < 5:
            continue

        shoulder_mid = (
            float((lsh[0] + rsh[0]) / 2.0) if lsh and rsh else float(coords[0][0]),
            float((lsh[1] + rsh[1]) / 2.0) if lsh and rsh else float(coords[0][1]),
        )
        hip_mid = (
            float((lhip[0] + rhip[0]) / 2.0) if lhip and rhip else float(coords[-1][0]),
            float((lhip[1] + rhip[1]) / 2.0) if lhip and rhip else float(coords[-1][1]),
        )
        torso_mid = (
            (shoulder_mid[0] + hip_mid[0]) / 2.0,
            (shoulder_mid[1] + hip_mid[1]) / 2.0,
        )

        torso_angle = abs(abs(angle_deg(hip_mid, shoulder_mid)) - 90.0)
        head_offset = abs((nose[0] if nose else shoulder_mid[0]) - shoulder_mid[0])
        shoulder_level = abs((rsh[1] if rsh else shoulder_mid[1]) - (lsh[1] if lsh else shoulder_mid[1]))
        hip_level = abs((rhip[1] if rhip else hip_mid[1]) - (lhip[1] if lhip else hip_mid[1]))
        seat_offset = abs(hip_mid[0] - saddle_mid[0]) if saddle_mid is not None else abs(hip_mid[0] - (w / 2.0))
        vertical_motion = abs(shoulder_mid[1] - hip_mid[1])
        horizontal_motion = abs(shoulder_mid[0] - hip_mid[0])
        left_knee_angle = compute_knee_angle(lhip, lknee, lankle) if lhip and lknee and lankle else None
        right_knee_angle = compute_knee_angle(rhip, rknee, rankle) if rhip and rknee and rankle else None
        left_ankle_angle = compute_knee_angle(lknee, lankle, lfoot) if lknee and lankle and lfoot else None
        right_ankle_angle = compute_knee_angle(rknee, rankle, rfoot) if rknee and rankle and rfoot else None
        sample_visibility = _safe_mean(
            [
                float(lms[pose_lm.NOSE].visibility),
                float(lms[pose_lm.LEFT_SHOULDER].visibility),
                float(lms[pose_lm.RIGHT_SHOULDER].visibility),
                float(lms[pose_lm.LEFT_HIP].visibility),
                float(lms[pose_lm.RIGHT_HIP].visibility),
                float(lms[pose_lm.LEFT_KNEE].visibility),
                float(lms[pose_lm.RIGHT_KNEE].visibility),
                float(lms[pose_lm.LEFT_ANKLE].visibility),
                float(lms[pose_lm.RIGHT_ANKLE].visibility),
            ]
        )

        sample = {
            "frame_index": int(idx),
            "time_sec": float(times[min(idx, len(times) - 1)]),
            "visibility": float(sample_visibility),
            "joints": {
                "nose": nose,
                "left_shoulder": lsh,
                "right_shoulder": rsh,
                "left_hip": lhip,
                "right_hip": rhip,
                "left_knee": lknee,
                "right_knee": rknee,
                "left_ankle": lankle,
                "right_ankle": rankle,
                "left_heel": lheel,
                "right_heel": rheel,
            },
            "torso_mid": torso_mid,
            "hip_mid": hip_mid,
            "angles": {
                "torso_angle_deg": float(torso_angle),
                "left_knee_angle_deg": float(left_knee_angle) if left_knee_angle is not None else None,
                "right_knee_angle_deg": float(right_knee_angle) if right_knee_angle is not None else None,
                "left_ankle_angle_deg": float(left_ankle_angle) if left_ankle_angle is not None else None,
                "right_ankle_angle_deg": float(right_ankle_angle) if right_ankle_angle is not None else None,
            },
            "measurements": {
                "head_offset_px": float(head_offset),
                "shoulder_level_px": float(shoulder_level),
                "hip_level_px": float(hip_level),
                "seat_offset_px": float(seat_offset),
                "vertical_motion_px": float(vertical_motion),
                "horizontal_motion_px": float(horizontal_motion),
            },
        }
        samples.append(sample)
        visibilities.append(float(sample_visibility))
        series["time"].append(float(sample["time_sec"]))
        series["torso_angle"].append(float(torso_angle))
        series["head_offset"].append(float(head_offset))
        series["shoulder_level"].append(float(shoulder_level))
        series["hip_level"].append(float(hip_level))
        series["seat_offset"].append(float(seat_offset))
        series["left_knee_angle"].append(float(left_knee_angle) if left_knee_angle is not None else float("nan"))
        series["right_knee_angle"].append(float(right_knee_angle) if right_knee_angle is not None else float("nan"))
        series["left_ankle_angle"].append(float(left_ankle_angle) if left_ankle_angle is not None else float("nan"))
        series["right_ankle_angle"].append(float(right_ankle_angle) if right_ankle_angle is not None else float("nan"))
        series["vertical_motion"].append(float(vertical_motion))
        series["horizontal_motion"].append(float(horizontal_motion))

    if not samples:
        return {
            "available": False,
            "estimated": True,
            "sample_count": 0,
            "confidence_pct": 0.0,
            "samples": [],
            "series": {},
            "summary": {},
            "notes": [
                "Pose landmarks were not reliable enough for rider joint analysis.",
                "Measurements that depend on rider posture will be estimated from the tracked saddle motion only.",
            ],
            "best_frame_index": None,
            "annotated_frame": "",
        }

    def _clean_series(values: List[float]) -> List[float]:
        return [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]

    clean_torso = _clean_series(series["torso_angle"])
    clean_head = _clean_series(series["head_offset"])
    clean_shoulder = _clean_series(series["shoulder_level"])
    clean_hip = _clean_series(series["hip_level"])
    clean_seat = _clean_series(series["seat_offset"])
    clean_vertical = _clean_series(series["vertical_motion"])
    clean_horizontal = _clean_series(series["horizontal_motion"])
    clean_left_knee = _clean_series(series["left_knee_angle"])
    clean_right_knee = _clean_series(series["right_knee_angle"])
    clean_left_ankle = _clean_series(series["left_ankle_angle"])
    clean_right_ankle = _clean_series(series["right_ankle_angle"])

    summary = {
        "available": True,
        "estimated": False,
        "sample_count": len(samples),
        "confidence_pct": float(max(0.0, min(100.0, _safe_mean(visibilities) * 100.0))),
        "torso_angle_mean_deg": _safe_mean(clean_torso),
        "torso_angle_std_deg": _safe_std(clean_torso),
        "head_alignment_mean_px": _safe_mean(clean_head),
        "head_alignment_std_px": _safe_std(clean_head),
        "shoulder_level_mean_px": _safe_mean(clean_shoulder),
        "shoulder_level_std_px": _safe_std(clean_shoulder),
        "hip_level_mean_px": _safe_mean(clean_hip),
        "hip_level_std_px": _safe_std(clean_hip),
        "seat_center_offset_px_mean": _safe_mean(clean_seat),
        "seat_center_offset_px_std": _safe_std(clean_seat),
        "vertical_motion_px_mean": _safe_mean(clean_vertical),
        "vertical_motion_px_std": _safe_std(clean_vertical),
        "horizontal_motion_px_mean": _safe_mean(clean_horizontal),
        "horizontal_motion_px_std": _safe_std(clean_horizontal),
        "left_knee_angle_mean_deg": _safe_mean(clean_left_knee),
        "left_knee_angle_std_deg": _safe_std(clean_left_knee),
        "right_knee_angle_mean_deg": _safe_mean(clean_right_knee),
        "right_knee_angle_std_deg": _safe_std(clean_right_knee),
        "left_ankle_angle_mean_deg": _safe_mean(clean_left_ankle),
        "left_ankle_angle_std_deg": _safe_std(clean_left_ankle),
        "right_ankle_angle_mean_deg": _safe_mean(clean_right_ankle),
        "right_ankle_angle_std_deg": _safe_std(clean_right_ankle),
        "visibility_mean": _safe_mean(visibilities),
        "visibility_std": _safe_std(visibilities),
    }

    best_sample = max(samples, key=lambda sample: float(sample.get("visibility", 0.0)))
    best_frame_index = int(best_sample.get("frame_index", 0))
    annotated = draw_annotation_overlay(
        frames[best_frame_index],
        points=points,
        pose_sample=best_sample,
        title="Rider & saddle reference frame",
    )

    notes = []
    if summary["confidence_pct"] < 45:
        notes.append("Pose detection confidence is limited; rider-specific scores are weighted conservatively.")
    if summary["sample_count"] < max_samples:
        notes.append("Fewer than the requested number of pose samples were usable.")

    return {
        "available": True,
        "estimated": False,
        "sample_count": len(samples),
        "confidence_pct": summary["confidence_pct"],
        "samples": samples,
        "series": series,
        "summary": summary,
        "notes": notes,
        "best_frame_index": best_frame_index,
        "annotated_frame": encode_frame_data_uri(annotated),
    }


def score_and_recommend(
    metrics: Dict[str, Any],
    horse_profile: str,
    saddle_type: str,
    discipline: str = "general_riding",
    pose_summary: Optional[Dict[str, Any]] = None,
) -> Dict:
    pose_summary = pose_summary or {}
    profile_key = normalize_horse_profile(horse_profile)
    discipline_key = normalize_discipline(discipline)
    profile_cfg = get_profile_config(profile_key)
    discipline_cfg = get_discipline_config(discipline_key)
    weights = discipline_cfg.get("weights", {}) or {"rider": 0.35, "horse": 0.20, "saddle": 0.25, "symmetry": 0.20}
    expected_rhythm = discipline_cfg.get("expected_rhythm_hz", (0.9, 1.8))
    expected_rhythm_low, expected_rhythm_high = float(expected_rhythm[0]), float(expected_rhythm[1])
    expected_rhythm_mid = (expected_rhythm_low + expected_rhythm_high) / 2.0

    pitch = float(metrics.get("pitch_mean_deg", 0.0))
    pitch_std = float(metrics.get("pitch_std_deg", 0.0))
    rock = float(metrics.get("rock_amplitude_deg", 0.0))
    drift_x = float(metrics.get("mid_drift_x_px", 0.0))
    drift_rate = float(metrics.get("mid_drift_rate_px_s", 0.0))
    bounce = float(metrics.get("mid_bounce_y_px", 0.0))
    shoulder_std = float(metrics.get("shoulder_level_std_px", 0.0))
    hip_std = float(metrics.get("hip_level_std_px", 0.0))
    cadence = float(metrics.get("cadence_hz", 0.0))
    clearance_collapse = bool(metrics.get("alignment", {}).get("clearance_collapse", False))
    clearance_min = float(metrics.get("alignment", {}).get("withers_clearance_min_px", 0.0))
    clearance_mean = float(metrics.get("alignment", {}).get("withers_clearance_mean_px", 0.0))
    bridging_proxy = float(metrics.get("alignment", {}).get("bridging_proxy", 0.0))
    topline_mean = float(metrics.get("alignment", {}).get("topline_mean_deg", 0.0))
    topline_std = float(metrics.get("alignment", {}).get("topline_std_deg", 0.0))
    saddle_topline_diff = float(metrics.get("alignment", {}).get("saddle_topline_diff_deg", 0.0))
    tracking_conf = float(metrics.get("tracking", {}).get("tracking_success_pct", 0.0))
    frame_w = float(metrics.get("tracking", {}).get("frame_w", 0.0) or 0.0)

    pose_conf = float(pose_summary.get("confidence_pct", 0.0) or 0.0)
    pose_visible = bool(pose_summary.get("available", False))
    pose_samples = int(pose_summary.get("sample_count", 0) or 0)
    pose_data = pose_summary.get("summary", {}) or {}

    torso_mean = float(pose_data.get("torso_angle_mean_deg", 0.0) or 0.0)
    torso_std = float(pose_data.get("torso_angle_std_deg", 0.0) or 0.0)
    head_offset = float(pose_data.get("head_alignment_mean_px", 0.0) or 0.0)
    head_offset_std = float(pose_data.get("head_alignment_std_px", 0.0) or 0.0)
    seat_offset = float(pose_data.get("seat_center_offset_px_mean", 0.0) or 0.0)
    seat_offset_std = float(pose_data.get("seat_center_offset_px_std", 0.0) or 0.0)
    vertical_motion = float(pose_data.get("vertical_motion_px_mean", 0.0) or 0.0)
    vertical_motion_std = float(pose_data.get("vertical_motion_px_std", 0.0) or 0.0)
    horizontal_motion = float(pose_data.get("horizontal_motion_px_mean", 0.0) or 0.0)
    horizontal_motion_std = float(pose_data.get("horizontal_motion_px_std", 0.0) or 0.0)
    left_knee_mean = float(pose_data.get("left_knee_angle_mean_deg", 0.0) or 0.0)
    right_knee_mean = float(pose_data.get("right_knee_angle_mean_deg", 0.0) or 0.0)
    left_knee_std = float(pose_data.get("left_knee_angle_std_deg", 0.0) or 0.0)
    right_knee_std = float(pose_data.get("right_knee_angle_std_deg", 0.0) or 0.0)
    left_ankle_mean = float(pose_data.get("left_ankle_angle_mean_deg", 0.0) or 0.0)
    right_ankle_mean = float(pose_data.get("right_ankle_angle_mean_deg", 0.0) or 0.0)
    left_ankle_std = float(pose_data.get("left_ankle_angle_std_deg", 0.0) or 0.0)
    right_ankle_std = float(pose_data.get("right_ankle_angle_std_deg", 0.0) or 0.0)

    profile_clearance_adjust = float(profile_cfg.get("clearance_threshold_adjust", 0.0) or 0.0)
    profile_rock_adjust = float(profile_cfg.get("rock_threshold_adjust", 0.0) or 0.0)
    profile_drift_adjust = float(profile_cfg.get("drift_threshold_adjust", 0.0) or 0.0)

    if frame_w <= 0:
        frame_w = 1920.0
    normalized = lambda value: abs(value) / frame_w * 100.0

    front_down_threshold = float(profile_cfg.get("front_down_threshold", -4.0) or -4.0)
    if discipline_key in {"dressage", "show_jumping", "equitation"}:
        front_down_threshold += 0.5
    if discipline_key in {"racing_gallop", "barrel_racing", "polo"}:
        front_down_threshold -= 0.5

    rock_threshold = 6.0 if saddle_type == "english" else 8.0
    rock_threshold += profile_rock_adjust
    if discipline_key in {"polo", "racing_gallop", "barrel_racing", "eventing"}:
        rock_threshold += 1.2

    drift_threshold = 35.0 + profile_drift_adjust
    if profile_key in {"round_barrel", "wide_build"}:
        drift_threshold += 8.0
    if discipline_key in {"polo", "barrel_racing", "racing_gallop"}:
        drift_threshold += 6.0

    align_threshold = 12.0
    bounce_threshold = 24.0
    seat_offset_threshold = 18.0

    def clamp(value: float) -> float:
        return max(0.0, min(100.0, value))

    def band(score: Optional[float]) -> str:
        return score_band(score if score is not None else None)

    flags: List[str] = []
    recs: List[str] = []

    if pitch < front_down_threshold:
        flags.append("Front-down saddle pitch detected.")
        recs.append("Check withers clearance and front balance. Use fitter guidance if the saddle appears to tip forward.")

    if rock > rock_threshold:
        flags.append("Higher saddle rocking detected across the ride.")
        recs.append("Check panel contact and girthing. Excessive rock can indicate bridging or unstable support.")

    if drift_x > drift_threshold:
        flags.append("Noticeable saddle drift detected.")
        recs.append("Review girth placement, pad grip, and saddle alignment to reduce movement.")

    if shoulder_std > align_threshold or hip_std > align_threshold:
        flags.append("Shoulder/hip alignment variance detected.")
        recs.append("Work toward even weight in both stirrups and a more level upper body.")

    if bounce > bounce_threshold:
        flags.append("Higher vertical bounce detected at the saddle midpoint.")
        recs.append("Use core engagement and a softer follow-through to reduce bounce.")

    if clearance_collapse:
        flags.append(f"Withers clearance risk detected (minimum {clearance_min:.1f}px).")
        recs.append("Check tree width, pad thickness, and flocking to preserve withers clearance.")

    if bridging_proxy > 4.0:
        flags.append("Rear/front clearance variation suggests bridging tendency.")
        recs.append("Evaluate panel contact and consider a fitter review if bridging persists.")

    if cadence == 0.0:
        recs.append("Cadence was not detected cleanly; use a longer, steadier straight-line clip for rhythm analysis.")
    elif cadence < expected_rhythm_low * 0.65:
        recs.append("Rhythm appears slower or more irregular than the discipline target.")

    seat_offset_norm = normalized(seat_offset)
    head_offset_norm = normalized(head_offset)
    torso_norm = abs(torso_mean)

    rider_posture = clamp(
        100.0
        - torso_norm * 3.0
        - pitch_std * 1.4
        - shoulder_std * 0.7
        - hip_std * 0.7
        - head_offset_norm * 0.5
    )
    rider_balance = clamp(
        100.0
        - drift_x * 0.35
        - abs(drift_rate) * 7.0
        - bounce * 1.3
        - seat_offset_norm * 0.8
    )
    rider_symmetry = clamp(
        100.0
        - abs(shoulder_std) * 1.8
        - abs(hip_std) * 1.8
        - abs(left_knee_mean - right_knee_mean) * 0.8
        - abs(left_ankle_mean - right_ankle_mean) * 0.8
        - head_offset_std * 0.15
    )
    rider_stability = clamp(
        100.0
        - torso_std * 3.5
        - vertical_motion_std * 0.03
        - horizontal_motion_std * 0.03
        - rock * 2.0
        - bounce * 1.0
    )

    target_knee = 150.0
    if discipline_key in {"show_jumping", "eventing", "hunter", "equitation"}:
        target_knee = 142.0
    elif discipline_key in {"racing_gallop", "barrel_racing", "polo"}:
        target_knee = 138.0
    elif discipline_key in {"dressage", "arena_riding"}:
        target_knee = 148.0

    knee_mean = _safe_mean([left_knee_mean, right_knee_mean])
    ankle_mean = _safe_mean([left_ankle_mean, right_ankle_mean])
    rider_leg_position = clamp(
        100.0
        - abs(knee_mean - target_knee) * 0.9
        - abs(ankle_mean - 160.0) * 0.45
        - max(left_knee_std, right_knee_std) * 0.6
        - max(left_ankle_std, right_ankle_std) * 0.4
        - seat_offset_norm * 0.3
    )
    rider_score = _score_value(
        rider_posture * 0.26
        + rider_balance * 0.26
        + rider_symmetry * 0.20
        + rider_stability * 0.18
        + rider_leg_position * 0.10
    )

    horse_movement = clamp(
        100.0
        - rock * 3.6
        - bounce * 1.4
        - abs(drift_rate) * 4.5
        - drift_x * 0.15
    )
    rhythm_penalty = abs(cadence - expected_rhythm_mid) / max(expected_rhythm_mid, 0.1)
    horse_rhythm = clamp(100.0 - rhythm_penalty * 70.0 - max(0.0, 1.0 - tracking_conf / 100.0) * 12.0)
    horse_consistency = clamp(
        100.0
        - pitch_std * 2.8
        - topline_std * 2.2
        - abs(saddle_topline_diff) * 1.2
        - abs(drift_rate) * 2.5
    )
    horse_symmetry = clamp(
        100.0
        - abs(shoulder_std + hip_std) * 1.3
        - topline_std * 1.5
        - abs(saddle_topline_diff) * 0.8
    )
    horse_topline = clamp(
        100.0
        - abs(topline_mean) * 1.2
        - topline_std * 2.0
        - max(0.0, 18.0 - clearance_mean) * 1.4
    )

    saddle_stability = clamp(
        100.0
        - rock * (5.2 if saddle_type == "english" else 4.6)
        - drift_x * 0.35
        - abs(pitch) * 1.8
        - (shoulder_std + hip_std) * 0.35
    )
    saddle_position = clamp(
        100.0
        - abs(pitch - profile_cfg.get("front_down_threshold", -4.0)) * 4.0
        - max(0.0, 16.0 - clearance_min - profile_clearance_adjust) * 2.2
        - drift_x * 0.15
    )
    saddle_balance = clamp(
        100.0
        - abs(saddle_topline_diff) * 1.8
        - bridging_proxy * 1.8
        - seat_offset_norm * 0.4
        - abs(drift_rate) * 3.0
    )

    discipline_score = clamp(
        rider_score * float(weights.get("rider", 0.35))
        + horse_movement * float(weights.get("horse", 0.20))
        + saddle_stability * float(weights.get("saddle", 0.25))
        + rider_symmetry * float(weights.get("symmetry", 0.20))
    )
    overall_score = clamp(
        rider_score * 0.30
        + horse_movement * 0.20
        + saddle_stability * 0.25
        + rider_symmetry * 0.10
        + discipline_score * 0.15
    )

    rider_level = "Beginner"
    if rider_score >= 75:
        rider_level = "Advanced"
    elif rider_score >= 50:
        rider_level = "Intermediate"

    fit_risk = "Low"
    if len(flags) == 1:
        fit_risk = "Medium"
    elif len(flags) >= 2:
        fit_risk = "High"

    coach_good = []
    if rider_posture >= 70:
        coach_good.append("Rider posture is generally controlled.")
    if rider_balance >= 70:
        coach_good.append("Balance stayed reasonably centered over the ride.")
    if horse_movement >= 70:
        coach_good.append("Horse movement remained comparatively consistent.")
    if saddle_stability >= 70:
        coach_good.append("Saddle movement stayed within a generally stable range.")

    coach_improve = []
    if rider_stability < 65:
        coach_improve.append("Reduce upper-body motion and keep the torso quieter through the stride.")
    if rider_leg_position < 65:
        coach_improve.append("Maintain a steadier lower leg and keep the heel quietly under the hip.")
    if horse_rhythm < 65:
        coach_improve.append("Use a more rhythmic straight-line clip if you want clearer cadence analysis.")
    if saddle_position < 65:
        coach_improve.append("Review saddle placement and withers clearance for a more centered position.")

    if not coach_improve:
        coach_improve.append("Continue repeating the same exercise with consistent rider posture and calm contact.")

    drills = [
        "Ride straight lines and check that shoulders stay level through the stride.",
        "Use light two-point or half-seat work to quiet the lower leg and reduce bounce.",
        "Repeat short intervals and compare the same metrics across sessions for trend tracking.",
    ]

    score_labels = {
        "overall": band(overall_score),
        "rider": band(rider_score),
        "horse_movement": band(horse_movement),
        "saddle_stability": band(saddle_stability),
        "symmetry": band(rider_symmetry),
        "discipline": band(discipline_score),
        "rider_posture": band(rider_posture),
        "rider_balance": band(rider_balance),
        "rider_symmetry": band(rider_symmetry),
        "rider_stability": band(rider_stability),
        "rider_leg_position": band(rider_leg_position),
        "horse_topline": band(horse_topline),
        "horse_rhythm": band(horse_rhythm),
        "horse_consistency": band(horse_consistency),
        "horse_symmetry": band(horse_symmetry),
        "saddle_position": band(saddle_position),
        "saddle_balance": band(saddle_balance),
    }

    return {
        "scores": {
            "overall": _score_value(overall_score),
            "rider": _score_value(rider_score),
            "rider_score": _score_value(rider_score),
            "horse_movement": _score_value(horse_movement),
            "horse_score": _score_value(horse_movement),
            "saddle_stability": _score_value(saddle_stability),
            "saddle_score": _score_value(saddle_stability),
            "symmetry": _score_value(rider_symmetry),
            "discipline": _score_value(discipline_score),
            "rider_posture": _score_value(rider_posture),
            "rider_balance": _score_value(rider_balance),
            "rider_symmetry": _score_value(rider_symmetry),
            "rider_stability": _score_value(rider_stability),
            "rider_leg_position": _score_value(rider_leg_position),
            "horse_topline": _score_value(horse_topline),
            "horse_rhythm": _score_value(horse_rhythm),
            "horse_consistency": _score_value(horse_consistency),
            "horse_symmetry": _score_value(horse_symmetry),
            "saddle_position": _score_value(saddle_position),
            "saddle_balance": _score_value(saddle_balance),
            "stability_label": band(saddle_stability),
            "fit_risk": fit_risk,
            "rider_level": rider_level,
            "labels": score_labels,
        },
        "tracking_confidence": tracking_conf,
        "flags": flags,
        "recommendations": recs
        + [
            f"Rider level estimate: {rider_level} ({rider_score}/100).",
        ],
        "coach": {
            "doing_well": coach_good,
            "to_improve": coach_improve,
            "drills": drills,
        },
        "context": {
            "horse_profile": profile_key,
            "saddle_type": saddle_type,
            "discipline": discipline_key,
            "horse_profile_label": str(profile_cfg.get("label", profile_key)),
            "discipline_label": str(discipline_cfg.get("label", discipline_key)),
            "discipline_focus": list(discipline_cfg.get("focus", []) or []),
            "discipline_notes": str(discipline_cfg.get("notes", "")),
            "expected_rhythm_hz": [expected_rhythm_low, expected_rhythm_high],
        },
        "pose_summary": pose_summary,
        "quality": {
            "tracking_success_pct": tracking_conf,
            "pose_confidence_pct": pose_conf,
            "pose_samples": pose_samples,
            "analysis_confidence_pct": float(max(0.0, min(100.0, tracking_conf * 0.55 + pose_conf * 0.45))),
            "pose_available": pose_visible,
            "pose_estimated": not pose_visible,
        },
    }


def evaluate_gear(gear: Dict[str, str], metrics: Dict[str, Any], horse_scheme: str) -> Dict[str, Any]:
    status = "PASS"
    notes: List[str] = []
    recs: List[str] = []
    helmet = gear.get("helmet", "yes")
    boots = gear.get("boots", "yes")

    if helmet != "yes":
        status = "FAIL"
        notes.append("Helmet missing.")
        recs.append("Wear a certified riding helmet.")
    if boots != "yes" and status != "FAIL":
        status = "WARN"
        notes.append("Riding boots not confirmed.")
        recs.append("Use heeled riding boots for safety.")

    drift = metrics.get("mid_drift_x_px", 0.0)
    rock = metrics.get("rock_amplitude_deg", 0.0)
    clearance = metrics.get("alignment", {}).get("withers_clearance_min_px", 0.0)

    if gear.get("breastplate", "no") == "no" and drift > 35:
        status = "WARN"
        notes.append("Consider breastplate for drift control.")
    if gear.get("pad", "normal") in ["gel", "shim"] and rock > 7:
        notes.append("Pad choice can affect rocking; verify fit with chosen pad.")
    if gear.get("girth", "standard") == "standard" and horse_scheme in ["round_barrel", "wide_build"] and drift > 30:
        recs.append("Try anatomical or mohair girth for better stability on wide barrels.")
    if clearance and clearance < 15:
        recs.append("Review pad thickness and tree width to improve withers clearance.")

    return {"status": status, "notes": notes, "recommendations": recs}


def compute_mark_scores(metrics: Dict[str, Any], scored: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    marks: Dict[str, Dict[str, Any]] = {}

    def clamp10(v: float) -> float:
        return max(0.0, min(10.0, v))

    stability10 = clamp10(scored["scores"]["saddle_stability"] / 10.0)
    marks["Stability"] = {"value": stability10, "improve": "Reduce rocking and drift to raise stability."}

    align_diff = abs(metrics.get("alignment", {}).get("saddle_topline_diff_deg", 0.0))
    align_score = clamp10(10.0 - min(10.0, align_diff / 2.0))
    marks["Alignment"] = {"value": align_score, "improve": "Balance saddle to match topline and keep even contact."}

    sym_std = metrics.get("symmetry", {}).get("rider", {}).get("shoulder_diff_std", 0.0) + metrics.get("symmetry", {}).get("rider", {}).get("hip_diff_std", 0.0)
    sym_score = clamp10(10.0 - min(10.0, sym_std / 5.0))
    marks["Symmetry"] = {"value": sym_score, "improve": "Level shoulders/hips and equalize stirrup pressure."}

    cadence = metrics.get("cadence_hz", 0.0)
    cadence_score = clamp10(10.0 - min(10.0, abs(cadence - 1.5) / 0.15 * 10.0))
    marks["Rhythm"] = {"value": cadence_score, "improve": "Keep steady tempo; use metronome or poles if needed."}

    clearance_min = metrics.get("alignment", {}).get("withers_clearance_min_px", 0.0)
    clearance_score = clamp10((clearance_min / max(1.0, clearance_min + 12.0)) * 10.0)
    marks["Clearance"] = {"value": clearance_score, "improve": "Maintain withers clearance; adjust pad/tree if low."}

    tracking_conf = metrics.get("tracking", {}).get("tracking_success_pct", 0.0)
    marks["Tracking"] = {"value": clamp10(tracking_conf / 10.0), "improve": "Film steady side view for higher tracking confidence."}

    bounce = metrics.get("mid_bounce_y_px", 0.0)
    bounce_score = clamp10(10.0 - min(10.0, bounce / 8.0))
    marks["Bounce"] = {"value": bounce_score, "improve": "Engage core and follow motion to reduce bounce."}

    drift = metrics.get("mid_drift_x_px", 0.0)
    drift_score = clamp10(10.0 - min(10.0, drift / 8.0))
    marks["Drift"] = {"value": drift_score, "improve": "Check girth, pad grip, and straightness to limit drift."}

    return marks


def build_simple_pdf_bytes(title: str, lines: List[str]) -> bytes:
    """
    Minimal PDF generator for fallback when weasyprint is unavailable.
    Produces a single-page PDF with text lines.
    """
    def esc(txt: str) -> str:
        return txt.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_lines = [
        "BT",
        "/F1 16 Tf",
        "72 760 Td",
        f"({esc(title)}) Tj",
        "/F1 11 Tf",
    ]
    for line in lines:
        content_lines.append("0 -16 Td")
        content_lines.append(f"({esc(line)}) Tj")
    content_lines.append("ET")
    content_lines.append("")
    content = "\n".join(content_lines)
    content_bytes = content.encode("latin-1", "replace")

    buf = io.BytesIO()

    def w(s: str):
        buf.write(s.encode("latin-1"))

    w("%PDF-1.4\n")
    obj_offsets: List[int] = []

    def obj(num: int, body: str):
        obj_offsets.append(buf.tell())
        w(f"{num} 0 obj\n{body}\nendobj\n")

    obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    obj(
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    )
    obj(4, f"<< /Length {len(content_bytes)} >>\nstream\n{content}\nendstream")
    obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_start = buf.tell()
    w(f"xref\n0 {len(obj_offsets) + 1}\n")
    w("0000000000 65535 f \n")
    for off in obj_offsets:
        w(f"{off:010d} 00000 n \n")
    w(f"trailer\n<< /Size {len(obj_offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF")
    return buf.getvalue()


def ensure_pdf_file(
    analysis_id: str,
    meta: dict,
    metrics: Dict[str, Any],
    scored: Dict[str, Any],
    mark_scores: Dict[str, Any],
    pdf_path: str,
) -> bool:
    pdf_file = storage.ensure_parent_dir(pdf_path)
    analysis_payload = scored.get("analysis_payload") or build_analysis_payload(
        analysis_id,
        meta,
        metrics,
        scored,
        mark_scores=mark_scores,
    )
    pdf_html = render_pdf_report_html(analysis_id, meta, metrics, scored, mark_scores, analysis_payload=analysis_payload)
    # Try full PDF via weasyprint; fallback to saving HTML so download never fails.
    pdf_renderer = get_weasyprint()
    if pdf_renderer is not None:
        try:
            pdf_renderer.HTML(string=pdf_html, base_url=BASE_DIR).write_pdf(os.fspath(pdf_file))
            return True
        except Exception:
            pass
    # Fallback: build a simple but valid PDF with key lines so it always opens.
    warnings_txt = "; ".join(scored.get("flags", [])) if scored.get("flags") else "None noted."
    rec_list = scored.get("recommendations", []) or []
    top_tip = rec_list[0] if len(rec_list) > 0 else "Keep a steady rhythm and balanced posture."
    tracking_pct = metrics.get("tracking", {}).get("tracking_success_pct", 0.0)
    frames_seen = int(metrics.get("tracking", {}).get("frames", metrics.get("frames_analyzed", 0)))
    gear_used = scored.get("gear_used", {}) or {}
    gear_detected = scored.get("gear_detection", {}) or {}
    gear_assessment = scored.get("gear_assessment", {}) or {}
    doing_well = scored.get("coach", {}).get("doing_well", []) or []
    to_improve = scored.get("coach", {}).get("to_improve", []) or []
    drills = scored.get("coach", {}).get("drills", []) or []
    align = metrics.get("alignment", {}) or {}
    sym_rider = metrics.get("symmetry", {}).get("rider", {}) if metrics.get("symmetry") else {}
    sym_saddle = metrics.get("symmetry", {}).get("saddle", {}) if metrics.get("symmetry") else {}
    sym_horse = metrics.get("symmetry", {}).get("horse", {}) if metrics.get("symmetry") else {}
    lines: List[str] = [
        f"ID: {analysis_id}",
        f"Scheme: {meta.get('horse_scheme','')} | Saddle: {meta.get('saddle_type','')}",
        f"Stability: {scored['scores'].get('saddle_stability',0)} ({scored['scores'].get('stability_label','')})",
        f"Fit risk: {scored['scores'].get('fit_risk','')}",
        f"Rider: {scored['scores'].get('rider_level','')} ({scored['scores'].get('rider_score',0)}/100)",
        f"Warnings: {warnings_txt}",
        f"Tracking: {tracking_pct:.1f}% over {frames_seen} frames",
        f"Pitch mean/std: {metrics.get('pitch_mean_deg',0.0):.2f}/{metrics.get('pitch_std_deg',0.0):.2f} deg",
        f"Rock amplitude: {metrics.get('rock_amplitude_deg',0.0):.2f} deg",
        f"Drift X: {metrics.get('mid_drift_x_px',0.0):.2f} px ({metrics.get('mid_drift_direction','stable')})",
        f"Bounce (px): {metrics.get('mid_bounce_y_px',0.0):.2f}; cadence: {metrics.get('cadence_hz',0.0):.2f} Hz; duration: {metrics.get('duration_sec',0.0):.2f}s",
        f"Withers clearance min/mean/std: {align.get('withers_clearance_min_px',0.0):.2f}/{align.get('withers_clearance_mean_px',0.0):.2f}/{align.get('withers_clearance_std_px',0.0):.2f} px",
        f"Topline mean/std: {align.get('topline_mean_deg',0.0):.2f}/{align.get('topline_std_deg',0.0):.2f} deg; saddle-topline diff: {align.get('saddle_topline_diff_deg',0.0):.2f} deg",
        f"Symmetry shoulders/hips (px std): {sym_rider.get('shoulder_diff_std',0.0):.2f}/{sym_rider.get('hip_diff_std',0.0):.2f}; roll proxy: {sym_saddle.get('roll_proxy',0.0):.2f}",
        f"Horse topline std: {sym_horse.get('topline_std',0.0):.2f}; confidence: {sym_horse.get('note','-')}",
        f"Gear status: {gear_assessment.get('status','') or 'PASS'}",
        "Gear used: "
        + "; ".join(
            [
                f"helmet {gear_used.get('helmet','-')}",
                f"boots {gear_used.get('boots','-')}",
                f"pad {gear_used.get('pad','-')}",
                f"girth {gear_used.get('girth','-')}",
                f"breastplate {gear_used.get('breastplate','-')}",
            ]
        ),
    ]
    if gear_detected.get("gear"):
        lines.append(
            "Detected: "
            + "; ".join([f"{k} {v} ({int(gear_detected.get('confidences', {}).get(k,0.0)*100)}%)" for k, v in gear_detected.get("gear", {}).items()])
        )
    if gear_detected.get("notes"):
        lines.append("Detection notes: " + " | ".join(gear_detected.get("notes", [])))
    if gear_assessment.get("notes"):
        lines.append("Gear notes: " + " | ".join(gear_assessment.get("notes", [])))
    if gear_assessment.get("recommendations"):
        lines.append("Gear recommendations: " + " | ".join(gear_assessment.get("recommendations", [])))
    if rec_list:
        lines.append("Recommendations: " + "; ".join(rec_list))
    if doing_well:
        lines.append("Doing well: " + "; ".join(doing_well))
    if to_improve:
        lines.append("To improve: " + "; ".join(to_improve))
    if drills:
        lines.append("Drills: " + "; ".join(drills))
    for name, data in mark_scores.items():
        lines.append(f"Mark {name}: {data.get('value',0):.1f}/10 - {data.get('improve','')}")
    lines.append(f"Top tip: {top_tip}")

    pdf_bytes = build_simple_pdf_bytes("Saddle Fit Report (lite)", lines)
    with open(pdf_file, "wb") as pf:
        pf.write(pdf_bytes)
    return True


def build_stick_svg(metrics: Dict[str, float], points: Optional[Dict[str, List[float]]] = None) -> str:
    # Use detected pose heights to warp the stick figure so it mirrors rider imbalance.
    shoulder_drop = 0.0
    hip_drop = 0.0
    if points:
        ls = points.get("left_shoulder")
        rs = points.get("right_shoulder")
        lh = points.get("left_hip")
        rh = points.get("right_hip")
        if ls and rs:
            shoulder_drop = float((rs[1] - ls[1]) * 0.25)
        if lh and rh:
            hip_drop = float((rh[1] - lh[1]) * 0.25)
    else:
        shoulder_drop = float(metrics.get("shoulder_level_std_px", 0.0) * 0.6)
        hip_drop = float(metrics.get("hip_level_std_px", 0.0) * 0.6)

    torso_lean = float(metrics.get("pitch_mean_deg", 0.0))
    torso_lean = max(-14.0, min(14.0, torso_lean))
    shoulder_drop = max(-18.0, min(18.0, shoulder_drop))
    hip_drop = max(-14.0, min(14.0, hip_drop))

    def figure(x_center: float, torso_tilt: float, shoulder_delta: float, hip_delta: float, label: str, color: str) -> str:
        head_y = 34.0
        arm_y = 74.0
        torso_top = (x_center, 64.0)
        torso_bottom = (x_center + torso_tilt, 130.0)
        leg_y = 184.0
        arm_span = 34.0
        leg_span = 24.0
        left_shoulder = (x_center - arm_span, arm_y - shoulder_delta * 0.5)
        right_shoulder = (x_center + arm_span, arm_y + shoulder_delta * 0.5)
        left_hip = (torso_bottom[0] - leg_span, torso_bottom[1] - hip_delta * 0.5)
        right_hip = (torso_bottom[0] + leg_span, torso_bottom[1] + hip_delta * 0.5)
        return (
            f"<circle cx='{x_center}' cy='{head_y}' r='12' fill='{color}' fill-opacity='0.12' stroke='{color}' stroke-width='3' />"
            f"<line x1='{torso_top[0]}' y1='{torso_top[1]}' x2='{torso_bottom[0]}' y2='{torso_bottom[1]}' stroke='{color}' stroke-width='6' stroke-linecap='round' />"
            f"<line x1='{left_shoulder[0]}' y1='{left_shoulder[1]}' x2='{right_shoulder[0]}' y2='{right_shoulder[1] - torso_tilt * 0.35}' stroke='{color}' stroke-width='5' stroke-linecap='round' />"
            f"<line x1='{torso_bottom[0]}' y1='{torso_bottom[1]}' x2='{left_hip[0]}' y2='{leg_y - hip_delta * 0.5}' stroke='{color}' stroke-width='5' stroke-linecap='round' />"
            f"<line x1='{torso_bottom[0]}' y1='{torso_bottom[1]}' x2='{right_hip[0]}' y2='{leg_y + hip_delta * 0.5}' stroke='{color}' stroke-width='5' stroke-linecap='round' />"
            f"<text x='{x_center}' y='{leg_y + 16}' fill='{color}' font-size='12' text-anchor='middle'>{label}</text>"
        )

    return (
        "<svg width='220' height='150' viewBox='0 0 380 214' xmlns='http://www.w3.org/2000/svg' style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:12px;'>"
        + figure(110, 0.0, 0.0, 0.0, "Balanced", "#22c55e")
        + figure(270, torso_lean * 0.8, shoulder_drop, hip_drop, "Detected", "#f59e0b")
        + "</svg>"
    )


def build_growth_svg(metrics: Dict[str, float], scores: Dict[str, float]) -> str:
    stability = float(scores.get("saddle_stability", 0.0))
    rider_score = float(scores.get("rider_score", 0.0))
    cadence_score = float(min(100.0, metrics.get("cadence_hz", 0.0) * 50.0))
    pitch_score = max(0.0, 100.0 - min(100.0, abs(metrics.get("pitch_mean_deg", 0.0)) * 10.0))
    rock_score = max(0.0, 100.0 - min(100.0, metrics.get("rock_amplitude_deg", 0.0) * 10.0))
    drift_score = max(0.0, 100.0 - min(100.0, metrics.get("mid_drift_x_px", 0.0) * 1.25))
    symmetry_score = max(
        0.0,
        100.0
        - min(100.0, ((metrics.get("shoulder_level_std_px", 0.0) + metrics.get("hip_level_std_px", 0.0)) / 30.0) * 100.0),
    )

    points = [
        ("Stability", stability),
        ("Rider", rider_score),
        ("Cadence", cadence_score),
        ("Pitch", pitch_score),
        ("Rock", rock_score),
        ("Drift", drift_score),
        ("Symmetry", symmetry_score),
    ]

    step_x = 45
    base_y = 140
    height = 170
    width = 60 + (len(points) - 1) * step_x + 60
    path_pts = []
    circles = []
    labels = []
    for i, (name, val) in enumerate(points):
        x = 40 + i * step_x
        y = base_y - (val / 100.0) * 120.0
        path_pts.append(f"{x},{y}")
        circles.append(f"<circle cx='{x}' cy='{y}' r='6' fill='#60a5fa' stroke='#0ea5e9' stroke-width='2' />")
        labels.append(f"<text x='{x}' y='{base_y + 18}' fill='#cbd5e1' font-size='11' text-anchor='middle'>{name}</text>")
        labels.append(f"<text x='{x}' y='{y - 10}' fill='#e5e7eb' font-size='11' text-anchor='middle'>{val:.0f}</text>")

    polyline = "<polyline fill='none' stroke='#38bdf8' stroke-width='3' points='" + " ".join(path_pts) + "' />"
    baseline = f"<line x1='24' y1='{base_y}' x2='{40 + (len(points)-1)*step_x + 20}' y2='{base_y}' stroke='rgba(255,255,255,0.15)' stroke-dasharray='4 6' />"

    return (
        f"<svg width='220' height='150' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:12px;'>"
        + baseline
        + polyline
        + "".join(circles)
        + "".join(labels)
        + "</svg>"
    )


def build_mark_chart(mark_scores: Dict[str, Dict[str, Any]]) -> str:
    items = list(mark_scores.items())
    height = 170
    bar_width = 32
    gap = 12
    width = 40 + len(items) * (bar_width + gap)
    bars = []
    labels = []
    for idx, (name, data) in enumerate(items):
        x = 20 + idx * (bar_width + gap)
        val = max(0.0, min(10.0, float(data.get("value", 0.0))))
        bar_h = (val / 10.0) * 120.0
        y = height - 28 - bar_h
        bars.append(f"<rect x='{x}' y='{y}' width='{bar_width}' height='{bar_h}' rx='6' fill='#38bdf8' />")
        labels.append(f"<text x='{x + bar_width/2}' y='{height - 10}' font-size='10' fill='#cbd5e1' text-anchor='middle'>{name}</text>")
        labels.append(f"<text x='{x + bar_width/2}' y='{y - 6}' font-size='10' fill='#e5e7eb' text-anchor='middle'>{val:.1f}</text>")
    return (
        f"<svg width='220' height='150' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:12px;'>"
        + "".join(bars)
        + "".join(labels)
        + "</svg>"
    )

def build_pdf_trend_svg(series: Dict[str, List[float]]) -> str:
    """
    Simple SVG line chart for PDF export (pitch and drift over time).
    """
    t = series.get("time") or []
    pitch = series.get("pitch") or []
    drift = series.get("drift_x") or []
    if len(t) == 0 or len(pitch) == 0 or len(drift) == 0:
        return "<div class='muted'>Trend data not available.</div>"

    pad_left, pad_right, pad_top, pad_bottom = 50, 40, 20, 40
    width, height = 520, 220
    w = width - pad_left - pad_right
    h = height - pad_top - pad_bottom

    time_start = t[0] or 0.0
    time_end = t[-1] or 1.0
    time_span = (time_end - time_start) or 1.0
    pitch_min, pitch_max = min(pitch), max(pitch)
    drift_min, drift_max = min(drift), max(drift)
    pitch_range = (pitch_max - pitch_min) or 1.0
    drift_range = (drift_max - drift_min) or 1.0

    def x_at(tv: float) -> float:
        return pad_left + ((tv - time_start) / time_span) * w

    def y_pitch(v: float) -> float:
        return pad_top + (1 - (v - pitch_min) / pitch_range) * h

    def y_drift(v: float) -> float:
        return pad_top + (1 - (v - drift_min) / drift_range) * h

    pitch_path = " ".join([f"{x_at(t[i]):.2f},{y_pitch(pitch[i]):.2f}" for i in range(len(pitch))])
    drift_path = " ".join([f"{x_at(t[i]):.2f},{y_drift(drift[i]):.2f}" for i in range(len(drift))])

    grid_lines = []
    for i in range(5):
        x = pad_left + (w * i / 4)
        grid_lines.append(f"<line x1='{x:.2f}' y1='{pad_top}' x2='{x:.2f}' y2='{pad_top + h}' stroke='rgba(0,0,0,0.1)' stroke-dasharray='4 4' />")
    for i in range(5):
        y = pad_top + (h * i / 4)
        grid_lines.append(f"<line x1='{pad_left}' y1='{y:.2f}' x2='{pad_left + w}' y2='{y:.2f}' stroke='rgba(0,0,0,0.1)' stroke-dasharray='4 4' />")

    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;'>"
        + "".join(grid_lines)
        + f"<polyline fill='none' stroke='#0ea5e9' stroke-width='2.5' points='{pitch_path}' />"
        + f"<polyline fill='none' stroke='#22c55e' stroke-width='2.5' points='{drift_path}' />"
        + f"<text x='{pad_left}' y='{pad_top - 4}' font-size='11' fill='#0f172a'>Pitch (deg)</text>"
        + f"<text x='{width - pad_right}' y='{pad_top - 4}' font-size='11' fill='#14532d' text-anchor='end'>Drift X (px)</text>"
        + f"<text x='{pad_left}' y='{height - 10}' font-size='11' fill='#475569'>time (sec) {time_start:.1f}-{time_end:.1f}</text>"
        + "</svg>"
    )


PROFESSIONAL_DISCLAIMER = (
    "This analysis provides visual and movement-based indicators and is not a substitute for an in-person assessment "
    "by a qualified saddle fitter, veterinarian, physiotherapist, or equine professional."
)


def _html_list(items: List[str], empty_msg: str = "No items available.") -> str:
    if not items:
        return f"<p class='muted'>{html_escape(empty_msg)}</p>"
    return "<ul class='list-tight'>" + "".join(f"<li>{html_escape(str(item))}</li>" for item in items) + "</ul>"


def _metric_rows(entries: List[Dict[str, Any]]) -> str:
    rows = []
    for entry in entries:
        name = html_escape(str(entry.get("name", "")))
        display = html_escape(str(entry.get("display", "N/A")))
        unit = html_escape(str(entry.get("unit", "")))
        status = html_escape(str(entry.get("status", "Insufficient Data")))
        source = html_escape(str(entry.get("source", "estimated")))
        note = html_escape(str(entry.get("note", "")))
        rows.append(
            f"<tr><td>{name}</td><td>{display}{(' ' + unit) if unit else ''}</td><td>{status}</td><td>{source}</td><td>{note}</td></tr>"
        )
    if not rows:
        return "<tr><td colspan='5'>No metrics available.</td></tr>"
    return "".join(rows)


def _metric_table_card(title: str, entries: List[Dict[str, Any]], empty_msg: str) -> str:
    if not entries:
        return f"""
        <div class="card">
          <h3 class="section-title">{html_escape(title)}</h3>
          <p class="muted">{html_escape(empty_msg)}</p>
        </div>
        """
    return f"""
    <div class="card">
      <h3 class="section-title">{html_escape(title)}</h3>
      <table>
        <tr><th>Metric</th><th>Value</th><th>Status</th><th>Source</th><th>Notes</th></tr>
        {_metric_rows(entries)}
      </table>
    </div>
    """


def _score_tiles(scores: Dict[str, Any]) -> str:
    tiles = [
        ("Overall Score", scores.get("overall", scores.get("rider_score", 0)), scores.get("labels", {}).get("overall", score_band(scores.get("overall"))), "Composite analysis score"),
        ("Rider Score", scores.get("rider", scores.get("rider_score", 0)), scores.get("labels", {}).get("rider", score_band(scores.get("rider"))), "Posture + balance + symmetry"),
        ("Horse Movement", scores.get("horse_movement", scores.get("horse_score", 0)), scores.get("labels", {}).get("horse_movement", score_band(scores.get("horse_movement"))), "Cadence + motion consistency"),
        ("Saddle Stability", scores.get("saddle_stability", scores.get("saddle_score", 0)), scores.get("labels", {}).get("saddle_stability", score_band(scores.get("saddle_stability"))), "Rocking + drift + balance"),
        ("Symmetry", scores.get("symmetry", 0), scores.get("labels", {}).get("symmetry", score_band(scores.get("symmetry"))), "Rider left/right alignment"),
    ]
    html_parts = []
    for label, value, band, caption in tiles:
        html_parts.append(
            f"""
            <div class="stat">
              <div class="k">{html_escape(label)}</div>
              <div class="v">{int(value) if value is not None else "N/A"}</div>
              <div class="muted">{html_escape(str(band))}</div>
              <div class="muted" style="margin-top:4px;">{html_escape(caption)}</div>
            </div>
            """
        )
    return "".join(html_parts)


def _comparison_direction(delta: Optional[float], higher_is_better: bool = True, threshold: float = COMPARISON_SIGNIFICANCE_THRESHOLD) -> str:
    if delta is None:
        return "No Significant Change"
    if abs(float(delta)) < float(threshold):
        return "No Significant Change"
    if higher_is_better:
        return "Improved" if float(delta) > 0 else "Declined"
    return "Improved" if float(delta) < 0 else "Declined"


def _comparison_row(metric: str, a: Optional[float], b: Optional[float], higher_is_better: bool = True, threshold: float = COMPARISON_SIGNIFICANCE_THRESHOLD, note: str = "") -> Dict[str, Any]:
    if a is None or b is None:
        return model_to_dict(
            ComparisonRow(
                metric=metric,
                ride_a=a,
                ride_b=b,
                delta=None,
                percent_change=None,
                direction="No Significant Change",
                note=note or "Insufficient data for comparison.",
            )
        )
    delta = float(b) - float(a)
    percent_change = None if abs(float(a)) < 1e-6 else (delta / float(a)) * 100.0
    direction = _comparison_direction(delta, higher_is_better=higher_is_better, threshold=threshold)
    return model_to_dict(
        ComparisonRow(
            metric=metric,
            ride_a=float(a),
            ride_b=float(b),
            delta=float(delta),
            percent_change=float(percent_change) if percent_change is not None else None,
            direction=direction,
            note=note,
        )
    )


def _build_metric_collection(scores: Dict[str, Any], pose_summary: Dict[str, Any], metrics: Dict[str, Any], discipline_cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    pose_data = pose_summary.get("summary", {}) or {}
    pose_available = bool(pose_summary.get("available", False))
    pose_source = "measured" if pose_available else "estimated"
    raw_source = "measured"

    rider_metrics = [
        model_to_dict(make_metric_entry("Overall Rider Score", scores.get("rider"), "/100", score_band(scores.get("rider")), "measured", "Composite of posture, balance, symmetry, stability, and leg position.", 0)),
        model_to_dict(make_metric_entry("Rider Posture", scores.get("rider_posture"), "/100", score_band(scores.get("rider_posture")), pose_source, "Derived from torso alignment, head offset, and shoulder level variance.", 0)),
        model_to_dict(make_metric_entry("Rider Balance", scores.get("rider_balance"), "/100", score_band(scores.get("rider_balance")), pose_source, "Uses seat offset, drift, bounce, and upper-body control.", 0)),
        model_to_dict(make_metric_entry("Rider Symmetry", scores.get("rider_symmetry"), "/100", score_band(scores.get("rider_symmetry")), pose_source, "Uses left/right shoulder, hip, knee, and ankle balance.", 0)),
        model_to_dict(make_metric_entry("Rider Stability", scores.get("rider_stability"), "/100", score_band(scores.get("rider_stability")), pose_source, "Uses torso variability, vertical motion, and rocking.", 0)),
        model_to_dict(make_metric_entry("Rider Leg Position", scores.get("rider_leg_position"), "/100", score_band(scores.get("rider_leg_position")), pose_source, "Uses knee/ankle angles and lower-leg steadiness.", 0)),
        model_to_dict(make_metric_entry("Torso Angle", pose_data.get("torso_angle_mean_deg"), "deg", evidence_status(pose_available), pose_source, "Lower values mean a more vertical torso.", 1)),
        model_to_dict(make_metric_entry("Head Alignment", pose_data.get("head_alignment_mean_px"), "px", evidence_status(pose_available), pose_source, "Horizontal offset between the head and shoulder line.", 1)),
        model_to_dict(make_metric_entry("Seat Offset", pose_data.get("seat_center_offset_px_mean"), "px", evidence_status(pose_available), pose_source, "Horizontal offset from the saddle midpoint.", 1)),
        model_to_dict(make_metric_entry("Shoulder Level Variance", pose_data.get("shoulder_level_std_px"), "px", evidence_status(pose_available), pose_source, "How level the rider shoulders stay through time.", 1)),
        model_to_dict(make_metric_entry("Hip Level Variance", pose_data.get("hip_level_std_px"), "px", evidence_status(pose_available), pose_source, "How level the rider hips stay through time.", 1)),
        model_to_dict(make_metric_entry("Left Knee Angle", pose_data.get("left_knee_angle_mean_deg"), "deg", evidence_status(pose_available), pose_source, "Average left knee opening.", 1)),
        model_to_dict(make_metric_entry("Right Knee Angle", pose_data.get("right_knee_angle_mean_deg"), "deg", evidence_status(pose_available), pose_source, "Average right knee opening.", 1)),
    ]

    horse_metrics = [
        model_to_dict(make_metric_entry("Horse Movement Score", scores.get("horse_movement"), "/100", score_band(scores.get("horse_movement")), raw_source, "Composite of cadence, drift, rock amplitude, and motion consistency.", 0)),
        model_to_dict(make_metric_entry("Horse Rhythm", scores.get("horse_rhythm"), "/100", score_band(scores.get("horse_rhythm")), raw_source, "Rhythm compared with the expected range for the selected discipline.", 0)),
        model_to_dict(make_metric_entry("Horse Consistency", scores.get("horse_consistency"), "/100", score_band(scores.get("horse_consistency")), raw_source, "Measures repeatability of motion and topline stability.", 0)),
        model_to_dict(make_metric_entry("Horse Symmetry", scores.get("horse_symmetry"), "/100", score_band(scores.get("horse_symmetry")), raw_source, "Side-view proxy for left/right consistency and topline evenness.", 0)),
        model_to_dict(make_metric_entry("Horse Topline", scores.get("horse_topline"), "/100", score_band(scores.get("horse_topline")), raw_source, "Topline consistency and backline steadiness.", 0)),
        model_to_dict(make_metric_entry("Cadence", metrics.get("cadence_hz"), "Hz", evidence_status(True), raw_source, "Detected cadence from tracked motion.", 2)),
        model_to_dict(make_metric_entry("Rock Amplitude", metrics.get("rock_amplitude_deg"), "deg", evidence_status(True), raw_source, "Stride-to-stride rocking amplitude.", 2)),
        model_to_dict(make_metric_entry("Drift X", metrics.get("mid_drift_x_px"), "px", evidence_status(True), raw_source, "Horizontal saddle drift across the ride.", 2)),
    ]

    saddle_metrics = [
        model_to_dict(make_metric_entry("Saddle Stability", scores.get("saddle_stability"), "/100", score_band(scores.get("saddle_stability")), raw_source, "Overall saddle motion, rocking, and drift stability.", 0)),
        model_to_dict(make_metric_entry("Saddle Position", scores.get("saddle_position"), "/100", score_band(scores.get("saddle_position")), raw_source, "Centeredness and clearance balance.", 0)),
        model_to_dict(make_metric_entry("Rider-Saddle Balance", scores.get("saddle_balance"), "/100", score_band(scores.get("saddle_balance")), raw_source, "Relationship between rider position and saddle movement.", 0)),
        model_to_dict(make_metric_entry("Withers Clearance", metrics.get("alignment", {}).get("withers_clearance_min_px"), "px", evidence_status(True), raw_source, "Minimum visible clearance proxy.", 1)),
        model_to_dict(make_metric_entry("Bridging Proxy", metrics.get("alignment", {}).get("bridging_proxy"), "px", evidence_status(True), raw_source, "Front/rear clearance variation proxy.", 2)),
    ]

    discipline_priority = list(discipline_cfg.get("priority_metrics", []) or [])
    discipline_lookup = {
        "Rider Balance": ("rider_balance", "Balanced seat, drift, and bounce"),
        "Rider Posture": ("rider_posture", "Torso and upper-body control"),
        "Rider Symmetry": ("rider_symmetry", "Left/right consistency"),
        "Seat Stability": ("rider_stability", "Lower upper-body variation is better"),
        "Posture": ("rider_posture", "Torso and shoulder control"),
        "Horse Rhythm": ("horse_rhythm", "Cadence against the discipline target"),
        "Horse Movement": ("horse_movement", "Overall motion stability"),
        "Horse Consistency": ("horse_consistency", "Repeatability of motion"),
        "Horse Symmetry": ("horse_symmetry", "Side-view symmetry proxy"),
        "Saddle Stability": ("saddle_stability", "Rocking and drift control"),
        "Saddle Position": ("saddle_position", "Centeredness and clearance"),
        "Saddle Balance": ("saddle_balance", "Rider-saddle relationship"),
        "Lower-Leg Stability": ("rider_leg_position", "Leg quietness and consistency"),
        "Hip Angle": ("rider_leg_position", "Target hip angle varies by discipline"),
        "Knee Angle": ("rider_leg_position", "Target knee angle varies by discipline"),
        "Dynamic Balance": ("rider_balance", "Movement under dynamic work"),
    }
    discipline_metrics: List[Dict[str, Any]] = []
    for label in discipline_priority[:5]:
        score_key, note = discipline_lookup.get(label, (None, discipline_cfg.get("notes", "")))
        if score_key and score_key in scores:
            discipline_metrics.append(
                model_to_dict(
                    make_metric_entry(
                        label,
                        scores.get(score_key),
                        "/100",
                        score_band(scores.get(score_key)),
                        pose_source if score_key.startswith("rider") else raw_source,
                        note,
                        0,
                    )
                )
            )
        else:
            discipline_metrics.append(
                model_to_dict(
                    make_metric_entry(
                        label,
                        None,
                        "/100",
                        "Insufficient Data",
                        "insufficient",
                        str(discipline_cfg.get("notes", "")),
                        0,
                    )
                )
            )

    return {
        "rider_metrics": rider_metrics,
        "horse_metrics": horse_metrics,
        "saddle_metrics": saddle_metrics,
        "discipline_metrics": discipline_metrics,
    }


def build_analysis_payload(
    analysis_id: str,
    meta: Dict[str, Any],
    metrics: Dict[str, Any],
    scored: Dict[str, Any],
    points: Optional[Dict[str, List[float]]] = None,
    pose_summary: Optional[Dict[str, Any]] = None,
    mark_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    horse_profile, discipline, saddle_type = resolve_analysis_context(meta)
    profile_cfg = get_profile_config(horse_profile)
    discipline_cfg = get_discipline_config(discipline)
    pose_summary = pose_summary or scored.get("pose_summary", {}) or {}
    mark_scores = mark_scores or compute_mark_scores(metrics, scored)
    scores = scored.get("scores", {}) or {}
    metric_groups = _build_metric_collection(scores, pose_summary, metrics, discipline_cfg)
    quality = dict(scored.get("quality", {}) or {})
    quality.setdefault("tracking_success_pct", float(metrics.get("tracking", {}).get("tracking_success_pct", 0.0) or 0.0))
    quality.setdefault("pose_confidence_pct", float(pose_summary.get("confidence_pct", 0.0) or 0.0))
    quality.setdefault("analysis_confidence_pct", float(max(0.0, min(100.0, quality.get("tracking_success_pct", 0.0) * 0.55 + quality.get("pose_confidence_pct", 0.0) * 0.45))))
    quality.setdefault("frames_analyzed", int(metrics.get("frames_analyzed", 0) or 0))
    quality.setdefault("duration_sec", float(metrics.get("duration_sec", 0.0) or 0.0))
    quality.setdefault("missing_data", [])
    if not pose_summary.get("available", False):
        quality["missing_data"].append("Rider joint angles could not be measured reliably.")
    if float(metrics.get("cadence_hz", 0.0) or 0.0) <= 0.0:
        quality["missing_data"].append("Cadence could not be inferred from the available motion.")

    strengths: List[str] = []
    improvement: List[str] = []
    score_pairs = [
        ("Overall analysis", scores.get("overall", 0)),
        ("Rider posture", scores.get("rider_posture", 0)),
        ("Rider balance", scores.get("rider_balance", 0)),
        ("Rider symmetry", scores.get("rider_symmetry", 0)),
        ("Rider stability", scores.get("rider_stability", 0)),
        ("Rider leg position", scores.get("rider_leg_position", 0)),
        ("Horse movement", scores.get("horse_movement", 0)),
        ("Horse rhythm", scores.get("horse_rhythm", 0)),
        ("Horse consistency", scores.get("horse_consistency", 0)),
        ("Horse symmetry", scores.get("horse_symmetry", 0)),
        ("Saddle stability", scores.get("saddle_stability", 0)),
        ("Saddle position", scores.get("saddle_position", 0)),
        ("Rider-saddle balance", scores.get("saddle_balance", 0)),
    ]
    for label, value in score_pairs:
        if value >= 75:
            strengths.append(f"{label}: {int(value)}/100")
        elif value < 60:
            improvement.append(f"{label}: {int(value)}/100")

    recommendations: List[str] = []
    for item in scored.get("recommendations", []) or []:
        if item not in recommendations:
            recommendations.append(item)
    for item in scored.get("coach", {}).get("to_improve", []) or []:
        if item not in recommendations:
            recommendations.append(item)
    discipline_note = str(discipline_cfg.get("notes", ""))
    if discipline_note and discipline_note not in recommendations:
        recommendations.append(discipline_note)

    visual_evidence = {
        "annotated_frame": pose_summary.get("annotated_frame", ""),
        "best_frame_index": pose_summary.get("best_frame_index"),
        "track_chart": build_pdf_trend_svg(metrics.get("series", {}) or {}),
        "frame_count": int(metrics.get("frames_analyzed", 0) or 0),
        "points": points or {},
    }

    summary_cards = {
        "overall": scores.get("overall", 0),
        "rider": scores.get("rider", scores.get("rider_score", 0)),
        "horse_movement": scores.get("horse_movement", scores.get("horse_score", 0)),
        "saddle_stability": scores.get("saddle_stability", scores.get("saddle_score", 0)),
        "symmetry": scores.get("symmetry", 0),
        "discipline": scores.get("discipline", 0),
    }

    payload = {
        "analysis_id": analysis_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "video_metadata": model_to_dict(build_video_metadata(meta, metrics)),
        "horse_profile": horse_profile,
        "saddle_type": saddle_type,
        "discipline": discipline,
        "confidence": float(quality.get("analysis_confidence_pct", 0.0) or 0.0),
        "scores": scores,
        "rider_metrics": metric_groups["rider_metrics"],
        "horse_metrics": metric_groups["horse_metrics"],
        "saddle_metrics": metric_groups["saddle_metrics"],
        "discipline_metrics": metric_groups["discipline_metrics"],
        "strengths": strengths,
        "areas_for_improvement": improvement,
        "recommendations": recommendations,
        "visual_evidence": visual_evidence,
        "quality": quality,
        "disclaimer": PROFESSIONAL_DISCLAIMER,
        "report_url": f"/report/{analysis_id}",
        "pdf_url": f"/report/{analysis_id}.pdf",
        "horse_profile_label": str(profile_cfg.get("label", horse_profile)),
        "discipline_label": str(discipline_cfg.get("label", discipline)),
        "discipline_focus": list(discipline_cfg.get("focus", []) or []),
        "discipline_notes": str(discipline_cfg.get("notes", "")),
        "summary_cards": summary_cards,
        "mark_scores": mark_scores,
        "metrics": metrics,
        "pose_summary": pose_summary,
        "calibration_points": points or {},
        "points": points or {},
        "scores_alias": {
            "overall": scores.get("overall", 0),
            "rider_score": scores.get("rider", scores.get("rider_score", 0)),
            "horse_score": scores.get("horse_movement", scores.get("horse_score", 0)),
            "saddle_score": scores.get("saddle_stability", scores.get("saddle_score", 0)),
        },
        "flags": scored.get("flags", []) or [],
        "coach": scored.get("coach", {}) or {},
        "gear_assessment": scored.get("gear_assessment", {}) or {},
        "gear_detection": scored.get("gear_detection", {}) or {},
        "gear_used": scored.get("gear_used", {}) or {},
        "analysis_sections_html": "",
    }
    payload["analysis_sections_html"] = build_analysis_sections_html(payload)
    return payload


def build_analysis_sections_html(analysis: Dict[str, Any]) -> str:
    scores = analysis.get("scores", {}) or {}
    metrics = analysis.get("metrics", {}) or {}
    pose_summary = analysis.get("pose_summary", {}) or {}
    quality = analysis.get("quality", {}) or {}
    video_meta = analysis.get("video_metadata", {}) or {}

    executive_summary = (
        f"Overall score {int(scores.get('overall', 0))} with rider score {int(scores.get('rider', scores.get('rider_score', 0)))}. "
        f"Horse movement scored {int(scores.get('horse_movement', scores.get('horse_score', 0)))} and saddle stability scored {int(scores.get('saddle_stability', scores.get('saddle_score', 0)))}. "
        f"Discipline focus: {analysis.get('discipline_label', analysis.get('discipline', 'General Riding'))}."
    )

    rider_entries = analysis.get("rider_metrics", []) or []
    horse_entries = analysis.get("horse_metrics", []) or []
    saddle_entries = analysis.get("saddle_metrics", []) or []
    discipline_entries = analysis.get("discipline_metrics", []) or []
    rider_posture_entries = [e for e in rider_entries if e.get("name") in {"Rider Posture", "Torso Angle", "Head Alignment", "Shoulder Level Variance", "Hip Level Variance"}]
    rider_balance_entries = [e for e in rider_entries if e.get("name") in {"Rider Balance", "Seat Offset", "Head Alignment"}]
    rider_joint_entries = [e for e in rider_entries if e.get("name") in {"Left Knee Angle", "Right Knee Angle", "Torso Angle"}]
    rider_stability_entries = [e for e in rider_entries if e.get("name") in {"Rider Stability", "Shoulder Level Variance", "Hip Level Variance", "Rider Symmetry", "Rider Leg Position"}]
    horse_movement_entries = [e for e in horse_entries if e.get("name") in {"Horse Movement Score", "Horse Rhythm", "Horse Consistency", "Cadence", "Rock Amplitude", "Drift X"}]
    horse_symmetry_entries = [e for e in horse_entries if e.get("name") in {"Horse Symmetry", "Horse Topline"}]
    saddle_analysis_entries = [e for e in saddle_entries if e.get("name") in {"Saddle Stability", "Saddle Position", "Rider-Saddle Balance", "Withers Clearance", "Bridging Proxy"}]

    sections = [
        f"""
        <div class="card">
          <h3 class="section-title">Executive Summary</h3>
          <p>{html_escape(executive_summary)}</p>
          <div class="grid">
            {_score_tiles(scores)}
          </div>
          <div class="grid two" style="margin-top:12px;">
            <div>
              <div class="k">Video</div>
              <p class="muted">{html_escape(str(video_meta.get("filename", "")))}</p>
              <p class="muted">Frames analyzed: {int(metrics.get("frames_analyzed", 0) or 0)} | Confidence: {float(quality.get("analysis_confidence_pct", analysis.get("confidence", 0.0))):.1f}%</p>
            </div>
            <div>
              <div class="k">Discipline context</div>
              <p class="muted">{html_escape(str(analysis.get("discipline_notes", "")))}</p>
            </div>
          </div>
        </div>
        """,
        _metric_table_card("Rider Posture Analysis", rider_posture_entries, "Not enough pose confidence to calculate rider posture reliably."),
        _metric_table_card("Rider Balance Analysis", rider_balance_entries, "Not enough pose confidence to calculate rider balance reliably."),
        _metric_table_card("Rider Joint Angles", rider_joint_entries, "Joint angles are not available from the current video angle."),
        _metric_table_card("Rider Stability Analysis", rider_stability_entries, "Rider stability could not be measured with enough confidence."),
        _metric_table_card("Horse Movement Analysis", horse_movement_entries, "Horse movement metrics were not available from the extracted motion."),
        _metric_table_card("Horse Symmetry Analysis", horse_symmetry_entries, "Horse symmetry is limited by the side-view camera perspective."),
        _metric_table_card("Saddle Stability Analysis", saddle_analysis_entries, "Saddle stability indicators were not strong enough to score confidently."),
        _metric_table_card("Discipline-Specific Analysis", discipline_entries, "Discipline-specific metrics were not available."),
        f"""
        <div class="card">
          <h3 class="section-title">Visual Evidence</h3>
          <div class="grid two">
            <div>
              <img src="{html_escape(str(analysis.get('visual_evidence', {}).get('annotated_frame', '')))}" alt="Annotated analysis frame" style="width:100%; border-radius:14px; border:1px solid #e2e8f0; background:#0f172a;" />
              <p class="muted" style="margin-top:8px;">Annotated representative frame with detected saddle and rider reference points.</p>
            </div>
            <div>
              <div class="viz">{build_pdf_trend_svg(metrics.get("series", {}) or {})}</div>
              <p class="muted" style="margin-top:8px;">Motion trend summary from the extracted tracking series.</p>
            </div>
          </div>
        </div>
        """,
        f"""
        <div class="card">
          <h3 class="section-title">Key Strengths</h3>
          {_html_list(analysis.get("strengths", []) or [], "No strong areas were isolated with enough confidence.")}
        </div>
        """,
        f"""
        <div class="card">
          <h3 class="section-title">Areas for Improvement</h3>
          {_html_list(analysis.get("areas_for_improvement", []) or [], "No major areas for improvement were isolated.")}
        </div>
        """,
        f"""
        <div class="card">
          <h3 class="section-title">Recommendations</h3>
          {_html_list(analysis.get("recommendations", []) or [], "No actionable recommendations were generated.")}
        </div>
        """,
        f"""
        <div class="card">
          <h3 class="section-title">Data Quality / Confidence</h3>
          <table>
            <tr><td>Tracking success</td><td>{float(quality.get('tracking_success_pct', metrics.get('tracking', {}).get('tracking_success_pct', 0.0))):.1f}%</td></tr>
            <tr><td>Pose confidence</td><td>{float(quality.get('pose_confidence_pct', pose_summary.get('confidence_pct', 0.0))):.1f}%</td></tr>
            <tr><td>Analysis confidence</td><td>{float(quality.get('analysis_confidence_pct', analysis.get('confidence', 0.0))):.1f}%</td></tr>
            <tr><td>Pose samples</td><td>{int(quality.get('pose_samples', pose_summary.get('sample_count', 0)) or 0)}</td></tr>
            <tr><td>Frames analyzed</td><td>{int(quality.get('frames_analyzed', metrics.get('frames_analyzed', 0)) or 0)}</td></tr>
            <tr><td>Missing data</td><td>{html_escape("; ".join(quality.get('missing_data', []) or ["None"]))}</td></tr>
          </table>
        </div>
        """,
        f"""
        <div class="card">
          <h3 class="section-title">Professional Disclaimer</h3>
          <p class="muted">{html_escape(str(analysis.get("disclaimer", PROFESSIONAL_DISCLAIMER)))}</p>
        </div>
        """,
    ]
    return "\n".join(sections)


def build_comparison_payload(
    comparison_id: str,
    meta: Dict[str, Any],
    label_a: str,
    label_b: str,
    analysis_a: Dict[str, Any],
    analysis_b: Dict[str, Any],
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    horse_profile, discipline, saddle_type = resolve_analysis_context(meta)
    scores_a = analysis_a.get("scores", {}) or {}
    scores_b = analysis_b.get("scores", {}) or {}
    metrics_a = analysis_a.get("metrics", {}) or {}
    metrics_b = analysis_b.get("metrics", {}) or {}
    quality_a = analysis_a.get("quality", {}) or {}
    quality_b = analysis_b.get("quality", {}) or {}
    discipline_cfg = get_discipline_config(discipline)

    comparison_rows = [
        _comparison_row("Overall Score", scores_a.get("overall", 0), scores_b.get("overall", 0), True, note="Composite score across rider, horse, saddle, and discipline."),
        _comparison_row("Rider Score", scores_a.get("rider", scores_a.get("rider_score", 0)), scores_b.get("rider", scores_b.get("rider_score", 0)), True, note="Posture, balance, symmetry, stability, and leg position."),
        _comparison_row("Horse Movement", scores_a.get("horse_movement", scores_a.get("horse_score", 0)), scores_b.get("horse_movement", scores_b.get("horse_score", 0)), True, note="Motion consistency and cadence proxy."),
        _comparison_row("Saddle Stability", scores_a.get("saddle_stability", scores_a.get("saddle_score", 0)), scores_b.get("saddle_stability", scores_b.get("saddle_score", 0)), True, note="Rocking and drift stability."),
        _comparison_row("Symmetry", scores_a.get("symmetry", 0), scores_b.get("symmetry", 0), True, note="Left/right rider and horse balance."),
        _comparison_row("Discipline Score", scores_a.get("discipline", 0), scores_b.get("discipline", 0), True, note=str(discipline_cfg.get("notes", ""))),
        _comparison_row("Rider Posture", scores_a.get("rider_posture", 0), scores_b.get("rider_posture", 0), True),
        _comparison_row("Rider Balance", scores_a.get("rider_balance", 0), scores_b.get("rider_balance", 0), True),
        _comparison_row("Rider Symmetry", scores_a.get("rider_symmetry", 0), scores_b.get("rider_symmetry", 0), True),
        _comparison_row("Rider Stability", scores_a.get("rider_stability", 0), scores_b.get("rider_stability", 0), True),
        _comparison_row("Rider Leg Position", scores_a.get("rider_leg_position", 0), scores_b.get("rider_leg_position", 0), True),
        _comparison_row("Horse Rhythm", scores_a.get("horse_rhythm", 0), scores_b.get("horse_rhythm", 0), True),
        _comparison_row("Horse Consistency", scores_a.get("horse_consistency", 0), scores_b.get("horse_consistency", 0), True),
        _comparison_row("Horse Topline", scores_a.get("horse_topline", 0), scores_b.get("horse_topline", 0), True),
        _comparison_row("Horse Symmetry", scores_a.get("horse_symmetry", 0), scores_b.get("horse_symmetry", 0), True),
        _comparison_row("Saddle Position", scores_a.get("saddle_position", 0), scores_b.get("saddle_position", 0), True),
        _comparison_row("Rider-Saddle Balance", scores_a.get("saddle_balance", 0), scores_b.get("saddle_balance", 0), True),
        _comparison_row("Rock Amplitude", metrics_a.get("rock_amplitude_deg", 0.0), metrics_b.get("rock_amplitude_deg", 0.0), False, note="Lower rocking amplitude is better."),
        _comparison_row("Drift X", metrics_a.get("mid_drift_x_px", 0.0), metrics_b.get("mid_drift_x_px", 0.0), False, note="Lower drift means more stable alignment."),
        _comparison_row("Cadence", metrics_a.get("cadence_hz", 0.0), metrics_b.get("cadence_hz", 0.0), True, note="Compare against the discipline rhythm target."),
    ]

    stable_metrics = [row for row in comparison_rows if row["direction"] == "No Significant Change"]
    improved_metrics = [row for row in comparison_rows if row["direction"] == "Improved"]
    declined_metrics = [row for row in comparison_rows if row["direction"] == "Declined"]

    def summary_sentence() -> str:
        if scores_b.get("rider_stability", 0) >= scores_a.get("rider_stability", 0) + COMPARISON_SIGNIFICANCE_THRESHOLD:
            pieces = ["Ride B showed improved rider stability."]
        elif scores_b.get("rider_stability", 0) + COMPARISON_SIGNIFICANCE_THRESHOLD <= scores_a.get("rider_stability", 0):
            pieces = ["Ride B showed reduced rider stability."]
        else:
            pieces = []
        if scores_b.get("horse_movement", 0) >= scores_a.get("horse_movement", 0) + COMPARISON_SIGNIFICANCE_THRESHOLD:
            pieces.append("Horse movement consistency improved in Ride B.")
        if scores_b.get("saddle_stability", 0) >= scores_a.get("saddle_stability", 0) + COMPARISON_SIGNIFICANCE_THRESHOLD:
            pieces.append("Saddle stability was better in Ride B.")
        if not pieces:
            pieces.append("The rides were broadly similar across the measured metrics.")
        return " ".join(pieces)

    def key_metric_entry(label: str, value: Any, source: str = "measured") -> Dict[str, Any]:
        return model_to_dict(make_metric_entry(label, value, "/100", score_band(value if isinstance(value, (int, float)) else None), source, "", 0))

    comparison_payload = {
        "comparison_id": comparison_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "horse_profile": horse_profile,
        "saddle_type": saddle_type,
        "discipline": discipline,
        "overall_summary": summary_sentence(),
        "ride_a": {
            "analysis_id": analysis_a.get("analysis_id", ""),
            "video_metadata": analysis_a.get("video_metadata", {}),
            "horse_profile": analysis_a.get("horse_profile", horse_profile),
            "saddle_type": analysis_a.get("saddle_type", saddle_type),
            "discipline": analysis_a.get("discipline", discipline),
            "confidence": float(quality_a.get("analysis_confidence_pct", analysis_a.get("confidence", 0.0)) or 0.0),
            "scores": scores_a,
            "key_metrics": [
                key_metric_entry("Rider Score", scores_a.get("rider", scores_a.get("rider_score", 0))),
                key_metric_entry("Horse Movement", scores_a.get("horse_movement", scores_a.get("horse_score", 0))),
                key_metric_entry("Saddle Stability", scores_a.get("saddle_stability", scores_a.get("saddle_score", 0))),
                key_metric_entry("Symmetry", scores_a.get("symmetry", 0)),
            ],
        },
        "ride_b": {
            "analysis_id": analysis_b.get("analysis_id", ""),
            "video_metadata": analysis_b.get("video_metadata", {}),
            "horse_profile": analysis_b.get("horse_profile", horse_profile),
            "saddle_type": analysis_b.get("saddle_type", saddle_type),
            "discipline": analysis_b.get("discipline", discipline),
            "confidence": float(quality_b.get("analysis_confidence_pct", analysis_b.get("confidence", 0.0)) or 0.0),
            "scores": scores_b,
            "key_metrics": [
                key_metric_entry("Rider Score", scores_b.get("rider", scores_b.get("rider_score", 0))),
                key_metric_entry("Horse Movement", scores_b.get("horse_movement", scores_b.get("horse_score", 0))),
                key_metric_entry("Saddle Stability", scores_b.get("saddle_stability", scores_b.get("saddle_score", 0))),
                key_metric_entry("Symmetry", scores_b.get("symmetry", 0)),
            ],
        },
        "comparisons": comparison_rows,
        "strengths": [row["metric"] for row in improved_metrics[:5]],
        "areas_for_improvement": [row["metric"] for row in declined_metrics[:5]],
        "recommendations": [
            "Repeat the same test conditions if you want a cleaner apples-to-apples comparison.",
            "Prioritize the metrics that changed beyond the significance threshold.",
        ],
        "visual_evidence": {
            "ride_a_frame": analysis_a.get("visual_evidence", {}).get("annotated_frame", ""),
            "ride_b_frame": analysis_b.get("visual_evidence", {}).get("annotated_frame", ""),
            "ride_a_chart": analysis_a.get("visual_evidence", {}).get("track_chart", ""),
            "ride_b_chart": analysis_b.get("visual_evidence", {}).get("track_chart", ""),
        },
        "quality": {
            "ride_a_confidence_pct": float(quality_a.get("analysis_confidence_pct", analysis_a.get("confidence", 0.0)) or 0.0),
            "ride_b_confidence_pct": float(quality_b.get("analysis_confidence_pct", analysis_b.get("confidence", 0.0)) or 0.0),
            "ride_a_pose_confidence_pct": float(quality_a.get("pose_confidence_pct", analysis_a.get("pose_summary", {}).get("confidence_pct", 0.0)) or 0.0),
            "ride_b_pose_confidence_pct": float(quality_b.get("pose_confidence_pct", analysis_b.get("pose_summary", {}).get("confidence_pct", 0.0)) or 0.0),
            "ride_a_frames": int(metrics_a.get("frames_analyzed", 0) or 0),
            "ride_b_frames": int(metrics_b.get("frames_analyzed", 0) or 0),
        },
        "disclaimer": PROFESSIONAL_DISCLAIMER,
        "report_url": f"/compare_report/{comparison_id}",
        "pdf_url": f"/compare_report/{comparison_id}.pdf",
        "analysis_a": analysis_a,
        "analysis_b": analysis_b,
        "comparison_rows_table": comparison_rows,
        "stable_metrics": stable_metrics,
        "improved_metrics": improved_metrics,
        "declined_metrics": declined_metrics,
        "comparison_sections_html": "",
    }
    comparison_payload["comparison_sections_html"] = build_comparison_sections_html(comparison_payload)
    return comparison_payload


def build_comparison_sections_html(comparison: Dict[str, Any]) -> str:
    ride_a = comparison.get("ride_a", {}) or {}
    ride_b = comparison.get("ride_b", {}) or {}
    comparisons = comparison.get("comparisons", []) or []
    visual = comparison.get("visual_evidence", {}) or {}

    def rows_for(metric_names: List[str]) -> str:
        rows = []
        for row in comparisons:
            if row.get("metric") not in metric_names:
                continue
            rows.append(
                f"<tr><td>{html_escape(str(row.get('metric', '')))}</td><td>{row.get('ride_a', 'N/A')}</td><td>{row.get('ride_b', 'N/A')}</td><td>{row.get('delta', 'N/A')}</td><td>{html_escape(str(row.get('direction', 'No Significant Change')))}</td><td>{html_escape(str(row.get('note', '')))}</td></tr>"
            )
        if not rows:
            return "<tr><td colspan='6'>No comparison rows available.</td></tr>"
        return "".join(rows)

    def comp_card(title: str, metric_names: List[str]) -> str:
        return f"""
        <div class="card">
          <h3 class="section-title">{html_escape(title)}</h3>
          <table>
            <tr><th>Metric</th><th>Ride A</th><th>Ride B</th><th>Delta</th><th>Direction</th><th>Note</th></tr>
            {rows_for(metric_names)}
          </table>
        </div>
        """

    summary_cards = f"""
    <div class="grid">
      <div class="stat"><div class="k">Ride A overall</div><div class="v">{int(ride_a.get('scores', {}).get('overall', 0))}</div><div class="muted">{html_escape(score_band(ride_a.get('scores', {}).get('overall', 0)))}</div></div>
      <div class="stat"><div class="k">Ride B overall</div><div class="v">{int(ride_b.get('scores', {}).get('overall', 0))}</div><div class="muted">{html_escape(score_band(ride_b.get('scores', {}).get('overall', 0)))}</div></div>
      <div class="stat"><div class="k">Ride A rider</div><div class="v">{int(ride_a.get('scores', {}).get('rider', ride_a.get('scores', {}).get('rider_score', 0)))}</div><div class="muted">{html_escape(score_band(ride_a.get('scores', {}).get('rider', ride_a.get('scores', {}).get('rider_score', 0))))}</div></div>
      <div class="stat"><div class="k">Ride B rider</div><div class="v">{int(ride_b.get('scores', {}).get('rider', ride_b.get('scores', {}).get('rider_score', 0)))}</div><div class="muted">{html_escape(score_band(ride_b.get('scores', {}).get('rider', ride_b.get('scores', {}).get('rider_score', 0))))}</div></div>
    </div>
    """

    discipline_metric_names = ["Horse Rhythm", "Rider Balance", "Rider Posture", "Saddle Stability", "Rider Leg Position", "Horse Movement", "Horse Consistency", "Horse Topline"]
    return f"""
    <div class="card">
      <h3 class="section-title">Executive Comparison</h3>
      <p>{html_escape(str(comparison.get("overall_summary", "")))}</p>
      <div class="grid two">
        <div>
          <div class="k">Ride A</div>
          <p class="muted">Analysis {html_escape(str(ride_a.get("analysis_id", "")))} | Confidence {float(ride_a.get("confidence", 0.0)):.1f}%</p>
        </div>
        <div>
          <div class="k">Ride B</div>
          <p class="muted">Analysis {html_escape(str(ride_b.get("analysis_id", "")))} | Confidence {float(ride_b.get("confidence", 0.0)):.1f}%</p>
        </div>
      </div>
      {summary_cards}
    </div>
    <div class="card">
      <h3 class="section-title">Side-by-Side Visual Frames</h3>
      <div class="grid two">
        <div><img src="{html_escape(str(visual.get('ride_a_frame', '')))}" alt="Ride A annotated frame" style="width:100%; border-radius:14px; border:1px solid #e2e8f0;" /></div>
        <div><img src="{html_escape(str(visual.get('ride_b_frame', '')))}" alt="Ride B annotated frame" style="width:100%; border-radius:14px; border:1px solid #e2e8f0;" /></div>
      </div>
    </div>
    {comp_card("Rider Comparison", ["Overall Score", "Rider Score", "Rider Posture", "Rider Balance", "Rider Symmetry", "Rider Stability", "Rider Leg Position"])}
    {comp_card("Horse Comparison", ["Horse Movement", "Horse Rhythm", "Horse Consistency", "Horse Symmetry", "Horse Topline", "Cadence", "Rock Amplitude", "Drift X"])}
    {comp_card("Saddle Comparison", ["Saddle Stability", "Saddle Position", "Rider-Saddle Balance", "Withers Clearance", "Bridging Proxy"])}
    {comp_card("Discipline-Specific Comparison", discipline_metric_names)}
    <div class="card">
      <h3 class="section-title">Metric Difference Table</h3>
      <table>
        <tr><th>Metric</th><th>Ride A</th><th>Ride B</th><th>Delta</th><th>% Change</th><th>Direction</th><th>Note</th></tr>
        {''.join(f"<tr><td>{html_escape(str(row.get('metric','')))}</td><td>{row.get('ride_a', 'N/A')}</td><td>{row.get('ride_b', 'N/A')}</td><td>{row.get('delta', 'N/A')}</td><td>{row.get('percent_change', 'N/A')}</td><td>{html_escape(str(row.get('direction', 'No Significant Change')))}</td><td>{html_escape(str(row.get('note', '')))}</td></tr>" for row in comparisons)}
      </table>
    </div>
    <div class="card">
      <h3 class="section-title">Key Improvements</h3>
      {_html_list(comparison.get("strengths", []) or [], "No improvements exceeded the significance threshold.")}
    </div>
    <div class="card">
      <h3 class="section-title">Areas That Declined</h3>
      {_html_list(comparison.get("areas_for_improvement", []) or [], "No metrics declined beyond the threshold.")}
    </div>
    <div class="card">
      <h3 class="section-title">Stable Metrics</h3>
      {_html_list([row.get("metric", "") for row in comparison.get("stable_metrics", []) or []], "No stable metrics identified.")}
    </div>
    <div class="card">
      <h3 class="section-title">Recommended Next Steps</h3>
      {_html_list(comparison.get("recommendations", []) or [], "No recommendations available.")}
    </div>
    <div class="card">
      <h3 class="section-title">Discipline Notes</h3>
      <p class="muted">{html_escape(str(get_discipline_config(comparison.get('discipline', 'general_riding')).get('notes', '')))}</p>
    </div>
    """


def render_report_html(
    analysis_id: str,
    meta: dict,
    metrics: Dict[str, Any],
    scored: Dict,
    stick_svg: str,
    growth_svg: str,
    mark_chart_svg: str,
    pdf_available: bool,
    analysis_payload: Optional[Dict[str, Any]] = None,
    comparison_block: str = "",
    points: Optional[Dict[str, List[float]]] = None,
    mark_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    template = """
    <html>
    <head>
      <title>Saddle Fit Report - %%ANALYSIS_ID%%</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        body { font-family: 'Segoe UI', 'Inter', system-ui, sans-serif; margin:0; background:#f6f7fb; color:#0f172a; }
        .wrap { max-width: 1100px; margin: 0 auto; padding: 24px 16px 36px; }
        .hero { display:flex; justify-content: space-between; align-items: center; gap: 12px; background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:16px 18px; box-shadow:0 6px 18px rgba(15,23,42,0.06); }
        .card { background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:16px; box-shadow:0 4px 16px rgba(15,23,42,0.05); margin-top:14px; }
        .grid { display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
        .summary { display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top:14px; }
        .stat { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:12px 14px; }
        .k { color:#475569; font-size: 13px; }
        .v { font-size: 24px; font-weight: 800; color:#0f172a; }
        .pill { display:inline-block; padding: 6px 10px; border-radius: 999px; background: #ecfdf3; border: 1px solid #bbf7d0; color:#166534; font-size:12px; }
        .muted { color:#64748b; font-size: 13px; }
        a { color:#0ea5e9; }
        ul { margin: 8px 0 0 18px; }
        table { width:100%; border-collapse: collapse; margin-top: 8px; }
        td { padding: 10px 8px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }
        h1, h2, h3 { margin: 0; }
        .section-title { margin-bottom: 6px; font-weight: 700; color:#0f172a; }
        .two { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
        .list-tight li { margin-bottom:6px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="hero">
          <div>
            <h1>Riders Bay SaddleFit<br/>Professional Riding &amp; Saddle Analysis</h1>
            <p class="muted">Clear snapshot of rider posture, horse movement, saddle stability, discipline context, and gear safety.</p>
          </div>
          <div style="text-align:right;">
            <div class="pill">ID: %%ANALYSIS_ID%%</div>
            <div class="muted" style="margin-top:4px;">Horse: %%HORSE_PROFILE%% | Saddle: %%SADDLE%% | Discipline: %%DISCIPLINE%%</div>
            <div class="muted" style="margin-top:4px;">Confidence: %%CONFIDENCE%%%</div>
          </div>
        </div>

        <div class="summary">
          %%SUMMARY_TILES%%
        </div>

        <div class="card">
          <h3 class="section-title">Quick insights</h3>
          <ul class="list-tight">
            <li>Overall stability: <b>%%STABILITY_LABEL%%</b> (%%SADDLE_STABILITY%%)</li>
            <li>Fit risk: <b>%%FIT_RISK%%</b>; rider level: <b>%%RIDER_LEVEL%%</b> (%%RIDER_SCORE%%)</li>
            <li>Gear status: <b>%%GEAR_STATUS%%</b> - %%GEAR_HINT%%</li>
            <li>Warnings: %%FLAGS_TEXT%%</li>
            <li>Next step: %%NEXT_STEP%%</li>
          </ul>
        </div>

        <div class="grid two">
          <div class="card">
            <h3 class="section-title">Flags & recommendations</h3>
            <div class="k" style="margin-bottom:6px;">Flags</div>
            %%FLAGS_BLOCK%%
            <div class="k" style="margin-top:10px;">Overall recommendations</div>
            %%RECS_BLOCK%%
          </div>
          <div class="card">
            <h3 class="section-title">Coaching</h3>
            <div class="k" style="margin-bottom:6px;">Doing well</div>
            %%DOING_BLOCK%%
            <div class="k" style="margin-top:10px;">To improve</div>
            %%IMPROVE_BLOCK%%
            <div class="k" style="margin-top:10px;">Drills</div>
            %%DRILLS_BLOCK%%
          </div>
        </div>

        <div class="grid two">
          <div class="card">
            <h3 class="section-title">Key riding metrics</h3>
            <table>
              <tr><td>Pitch mean / std (deg)</td><td>%%PITCH_MEAN%% / %%PITCH_STD%%</td></tr>
              <tr><td>Rocking amplitude (deg)</td><td>%%ROCK%%</td></tr>
              <tr><td>Drift X (px + direction)</td><td>%%DRIFT%%</td></tr>
              <tr><td>Drift rate (px/s)</td><td>%%DRIFT_RATE%%</td></tr>
              <tr><td>Vertical bounce (px)</td><td>%%BOUNCE%%</td></tr>
              <tr><td>Cadence (Hz)</td><td>%%CADENCE%%</td></tr>
              <tr><td>Duration (sec)</td><td>%%DURATION%%</td></tr>
              <tr><td>Frames analyzed</td><td>%%FRAMES%%</td></tr>
            </table>
          </div>
          <div class="card">
            <h3 class="section-title">Alignment & symmetry</h3>
            <table>
              <tr><td>Topline mean / std (deg)</td><td>%%TOPLINE%%</td></tr>
              <tr><td>Saddle vs topline diff (deg)</td><td>%%SADDLE_DIFF%%</td></tr>
              <tr><td>Withers clearance min/mean/std (px)</td><td>%%CLEARANCE%%</td></tr>
              <tr><td>Clearance collapse</td><td>%%CLEARANCE_COLLAPSE%%</td></tr>
              <tr><td>Bridging proxy</td><td>%%BRIDGING%%</td></tr>
              <tr><td>Rider shoulder diff mean/std (px)</td><td>%%SHOULDER_DIFF%%</td></tr>
              <tr><td>Rider hip diff mean/std (px)</td><td>%%HIP_DIFF%%</td></tr>
              <tr><td>Saddle roll proxy</td><td>%%ROLL_PROXY%%</td></tr>
              <tr><td>Horse topline std (deg)</td><td>%%HORSE_TOPLINE_STD%%</td></tr>
              <tr><td>Confidence note</td><td>%%CONF_NOTE%%</td></tr>
            </table>
          </div>
        </div>

        <div class="card">
          <h3 class="section-title">Gear & safety</h3>
          <div class="grid">
            <div>
              <div class="k">Status</div>
              <div class="v">%%GEAR_STATUS%%</div>
              <div class="k" style="margin-top:6px;">Used for analysis</div>
              %%GEAR_USED_BLOCK%%
              <div class="k" style="margin-top:10px;">Gear notes</div>
              %%GEAR_NOTES_BLOCK%%
            </div>
            <div>
              <div class="k">Auto-detected from video</div>
              %%GEAR_DETECTED_BLOCK%%
              <div class="k" style="margin-top:10px;">Detection notes</div>
              %%DETECT_NOTES_BLOCK%%
              <div class="k" style="margin-top:10px;">Recommendations</div>
              %%GEAR_RECS_BLOCK%%
            </div>
          </div>
        </div>

        <div class="card">
          <h3 class="section-title">Marks & improvement</h3>
          <table>%%MARKS_ROWS%%</table>
        </div>

        %%COMPARISON_BLOCK%%

        %%ANALYSIS_SECTIONS%%

        <div class="grid">
          <div class="card">
            <h3 class="section-title">Visual aids</h3>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin-top:6px; align-items:flex-start;">
              <div><div class="k" style="margin-bottom:6px;">Rider posture</div><div style="max-width:260px;">%%STICK_SVG%%</div></div>
              <div><div class="k" style="margin-bottom:6px;">Progress snapshot</div><div style="max-width:260px;">%%GROWTH_SVG%%</div></div>
              <div><div class="k" style="margin-bottom:6px;">Marks by area</div><div style="max-width:260px;">%%MARK_CHART_SVG%%</div></div>
            </div>
          </div>
          <div class="card">
            <h3 class="section-title">Pitch & drift over time</h3>
            <canvas id="lineChart" width="960" height="300" style="width:100%; border-radius:12px; border:1px solid #e2e8f0; background:#f8fafc;"></canvas>
            <div class="k" style="margin-top:6px; display:flex; gap:12px; align-items:center;">
              <span style="display:flex; align-items:center; gap:6px;"><span style="width:14px; height:6px; background:#38bdf8; display:inline-block;"></span>Pitch</span>
              <span style="display:flex; align-items:center; gap:6px;"><span style="width:14px; height:6px; background:#22c55e; display:inline-block;"></span>Drift X</span>
            </div>
          </div>
        </div>

        <div class="grid two">
          <div class="card">
            <h3 class="section-title">Tracking quality</h3>
            <table>
              <tr><td>Tracking success (%)</td><td>%%TRACK_PCT%%%</td></tr>
              <tr><td>Frames processed</td><td>%%TRACK_FRAMES%%</td></tr>
              <tr><td>Reinitializations</td><td>%%REINITS%%</td></tr>
              <tr><td>Confidence note</td><td>Higher % = more reliable symmetry; side-view limits roll estimation.</td></tr>
            </table>
          </div>
          <div class="card">
            <h3 class="section-title">Open files</h3>
            <ul class="list-tight">
              %%PDF_ITEM%%
              <li><a href="/">Analyze another video</a></li>
            </ul>
          </div>
        </div>

        <div class="card">
          <h3 class="section-title">Instruction video</h3>
          <div class="muted" style="margin-bottom:6px;">Replay your ride (muted) while the assistant speaks the corrections.</div>
          <video id="reportVideo" controls muted playsinline style="width:100%; max-width:720px; height:320px; margin:0 auto; display:block; border-radius:12px; border:1px solid #e2e8f0; background:#000;">
            <source src="/video/%%ANALYSIS_ID%%" type="video/mp4">
            Your browser does not support video.
          </video>
          <button style="margin-top:12px; width:100%; background: linear-gradient(135deg,#22c55e,#06b6d4); border:none; border-radius:10px; padding:12px; color:#04120a; font-weight:800; cursor:pointer;" onclick="playCoach()">Play muted video + voice tips</button>
        </div>
      </div>

      <script>
        const recText = %%REC_JS%%;
        const seriesData = %%SERIES_JS%%;
        const pdfUrl = "/report_pdf/%%ANALYSIS_ID%%";

        function speak(text) {
          if (!('speechSynthesis' in window)) return;
          const u = new SpeechSynthesisUtterance(text);
          u.rate = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        }

        function playCoach() {
          const vid = document.getElementById("reportVideo");
          if (vid) {
            vid.muted = true; vid.defaultMuted = true; vid.volume = 0.0;
            try { vid.currentTime = 0; vid.play(); } catch (e) {}
            const cues = [
              "Check posture: straighten upper back, open chest.",
              "Drop your heels, keep steady contact.",
              "Maintain even shoulders, avoid drifting forward."
            ];
            let spoken = 0;
            vid.onended = () => spoken = cues.length;
            vid.ontimeupdate = () => {
              if (!('speechSynthesis' in window) || !vid.duration) return;
              const t = vid.currentTime / vid.duration;
              if (t > 0.2 && spoken === 0) { speak("Coach: " + cues[0]); spoken = 1; }
              if (t > 0.5 && spoken === 1) { speak("Coach: " + cues[1]); spoken = 2; }
              if (t > 0.8 && spoken === 2) { speak("Coach: " + cues[2] + " " + recText); spoken = 3; }
            };
            speak("Coach tips: " + recText);
          }
        }

        function downloadPdf() {
          const link = document.createElement("a");
          link.href = pdfUrl;
          link.download = "report_%%ANALYSIS_ID%%.pdf";
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }

        function drawLineChart() {
          const c = document.getElementById("lineChart");
          if (!c) return;
          const ctx = c.getContext("2d");
          ctx.clearRect(0, 0, c.width, c.height);

          const t = seriesData.time || [];
          const pitch = seriesData.pitch || [];
          const drift = seriesData.drift_x || [];
          if (!t.length || !pitch.length || !drift.length) return;

          const pad = { left: 60, right: 60, top: 24, bottom: 32 };
          const w = c.width - pad.left - pad.right;
          const h = c.height - pad.top - pad.bottom;
          const timeStart = t[0];
          const timeEnd = t[t.length - 1];
          const pitchMin = Math.min(...pitch);
          const pitchMax = Math.max(...pitch);
          const driftMin = Math.min(...drift);
          const driftMax = Math.max(...drift);
          const timeSpan = (timeEnd - timeStart) || 1;
          const pitchRange = (pitchMax - pitchMin) || 1;
          const driftRange = (driftMax - driftMin) || 1;

          const xAt = (timeVal) => pad.left + ((timeVal - timeStart) / timeSpan) * w;
          const yPitch = (val) => pad.top + (1 - (val - pitchMin) / pitchRange) * h;
          const yDrift = (val) => pad.top + (1 - (val - driftMin) / driftRange) * h;

          ctx.strokeStyle = "#e2e8f0";
          for (let i = 0; i <= 4; i++) {
            const y = pad.top + (h * i / 4);
            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
          }

          function drawSeries(data, yFn, color) {
            ctx.beginPath();
            data.forEach((v, i) => { const x = xAt(t[i] || timeStart); const y = yFn(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
            ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.stroke();
          }

          drawSeries(pitch, yPitch, "#38bdf8");
          drawSeries(drift, yDrift, "#22c55e");

          ctx.fillStyle = "#475569";
          ctx.font = "12px Arial";
          ctx.fillText("time (sec)", pad.left + w / 2 - 34, c.height - 10);
          ctx.fillText("pitch (deg)", pad.left - 56, pad.top - 10);
          ctx.fillText("drift X (px)", c.width - pad.right + 2, pad.top - 10);
        }

        window.addEventListener("load", () => { drawLineChart(); });
      </script>
    </body>
    </html>
    """

    rec_text = " ".join(scored.get("recommendations", [])) if scored.get("recommendations") else "Maintain balanced shoulders and hips, keep a steady cadence, and monitor saddle stability."
    rec_js = json.dumps(rec_text)
    points = points or {}
    mark_scores = mark_scores or {}
    series = metrics.get("series", {}) or {}
    series_js = json.dumps(series)
    if analysis_payload is None:
        analysis_payload = build_analysis_payload(analysis_id, meta, metrics, scored, points=points, pose_summary=scored.get("pose_summary", {}), mark_scores=mark_scores)
    summary_cards = analysis_payload.get("summary_cards", {}) or {}
    analysis_sections_html = analysis_payload.get("analysis_sections_html", "")
    horse_profile_label = analysis_payload.get("horse_profile_label", meta.get("horse_profile") or meta.get("horse_scheme", ""))
    discipline_label = analysis_payload.get("discipline_label", meta.get("discipline", "general_riding"))
    confidence_pct = float(analysis_payload.get("confidence", scored.get("quality", {}).get("analysis_confidence_pct", 0.0)) or 0.0)
    gear_detection = scored.get("gear_detection", {}) or meta.get("gear_detection", {}) or {}
    detected_gear = gear_detection.get("gear", {}) or {}
    gear_conf = gear_detection.get("confidences", {}) or {}
    gear_used = scored.get("gear_used", {}) or meta.get("gear_used", {}) or meta.get("gear", {}) or {}
    pdf_item = (
        "<li><a href='/report_pdf/" + analysis_id + "' download>Download PDF</a></li>"
        if pdf_available
        else "<li>PDF not available (install weasyprint and rerun).</li>"
    )

    def list_html(items, empty_msg: str) -> str:
        if not items:
            return f"<p class='muted'>{empty_msg}</p>"
        return "<ul class='list-tight'>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"

    def ol_html(items, empty_msg: str) -> str:
        if not items:
            return f"<p class='muted'>{empty_msg}</p>"
        return "<ol>" + "".join([f"<li>{x}</li>" for x in items]) + "</ol>"

    flags_text = "none noted" if len(scored.get("flags", [])) == 0 else "; ".join(scored["flags"])
    next_step = scored.get("recommendations", ["Keep a steady rhythm and balanced posture."])[0]
    gear_used_text = ", ".join([f"{k}:{v}" for k, v in gear_used.items()]) if gear_used else "Auto-detected"

    flags_block = list_html(scored.get("flags", []), "No issues flagged.")
    recs_block = list_html(scored.get("recommendations", []), "No recommendations generated.")
    doing_block = list_html(scored.get("coach", {}).get("doing_well", []), "No positives detected.")
    improve_block = list_html(scored.get("coach", {}).get("to_improve", []), "No improvement items.")
    drills_block = ol_html(scored.get("coach", {}).get("drills", []), "No drills provided.")

    gear_notes_block = list_html(scored.get("gear_assessment", {}).get("notes", []), "No gear notes.")
    gear_recs_block = list_html(scored.get("gear_assessment", {}).get("recommendations", []), "No gear recommendations.")
    gear_used_block = list_html([f"{k.title()}: {v}" for k, v in gear_used.items()], "No gear detected.")
    detected_list = [f"{k.title()}: {v} (conf {int(gear_conf.get(k, 0.0) * 100)}%)" for k, v in detected_gear.items()]
    gear_detected_block = list_html(detected_list, "No detection available.")
    detect_notes_block = list_html(gear_detection.get("notes", []), "No detection notes.")

    marks_rows = "".join([f"<tr><td>{name}</td><td>{data.get('value',0):.1f}/10</td><td>{data.get('improve','')}</td></tr>" for name, data in mark_scores.items()])

    replacements = {
        "ANALYSIS_ID": analysis_id,
        "SCHEME": meta.get("horse_scheme", ""),
        "HORSE_PROFILE": horse_profile_label,
        "SADDLE": meta.get("saddle_type", ""),
        "DISCIPLINE": discipline_label,
        "CONFIDENCE": f"{confidence_pct:.1f}",
        "SUMMARY_TILES": _score_tiles(summary_cards or scored.get("scores", {})),
        "SADDLE_STABILITY": f"{scored['scores']['saddle_stability']}/100",
        "STABILITY_LABEL": scored['scores']['stability_label'],
        "RIDER_LEVEL": scored['scores']['rider_level'],
        "RIDER_SCORE": f"{scored['scores']['rider_score']}/100",
        "FIT_RISK": scored['scores']['fit_risk'],
        "WARNINGS_COUNT": len(scored.get("flags", [])),
        "TRACK_PCT": f"{metrics.get('tracking', {}).get('tracking_success_pct',0.0):.1f}",
        "TRACK_FRAMES": int(metrics.get('tracking', {}).get('frames', metrics.get('frames_analyzed',0))),
        "GEAR_STATUS": scored.get("gear_assessment", {}).get("status", "PASS"),
        "GEAR_USED_TEXT": gear_used_text,
        "FLAGS_TEXT": flags_text,
        "NEXT_STEP": next_step,
        "FLAGS_BLOCK": flags_block,
        "RECS_BLOCK": recs_block,
        "DOING_BLOCK": doing_block,
        "IMPROVE_BLOCK": improve_block,
        "DRILLS_BLOCK": drills_block,
        "PITCH_MEAN": f"{metrics['pitch_mean_deg']:.2f}",
        "PITCH_STD": f"{metrics['pitch_std_deg']:.2f}",
        "ROCK": f"{metrics['rock_amplitude_deg']:.2f}",
        "DRIFT": f"{metrics['mid_drift_x_px']:.2f} ({metrics.get('mid_drift_direction','stable')})",
        "DRIFT_RATE": f"{metrics.get('mid_drift_rate_px_s',0.0):.2f}",
        "BOUNCE": f"{metrics.get('mid_bounce_y_px',0.0):.2f}",
        "CADENCE": f"{metrics['cadence_hz']:.2f}",
        "DURATION": f"{metrics.get('duration_sec',0.0):.2f}",
        "FRAMES": int(metrics.get('frames_analyzed',0)),
        "TOPLINE": f"{metrics['alignment']['topline_mean_deg']:.2f} / {metrics['alignment']['topline_std_deg']:.2f}",
        "SADDLE_DIFF": f"{metrics['alignment']['saddle_topline_diff_deg']:.2f}",
        "CLEARANCE": f"{metrics['alignment']['withers_clearance_min_px']:.2f} / {metrics['alignment']['withers_clearance_mean_px']:.2f} / {metrics['alignment']['withers_clearance_std_px']:.2f}",
        "CLEARANCE_COLLAPSE": "Yes" if metrics['alignment']['clearance_collapse'] else "No",
        "BRIDGING": f"{metrics['alignment']['bridging_proxy']:.2f}",
        "SHOULDER_DIFF": f"{metrics['symmetry']['rider']['shoulder_diff_mean']:.2f} / {metrics['symmetry']['rider']['shoulder_diff_std']:.2f}",
        "HIP_DIFF": f"{metrics['symmetry']['rider']['hip_diff_mean']:.2f} / {metrics['symmetry']['rider']['hip_diff_std']:.2f}",
        "ROLL_PROXY": f"{metrics['symmetry']['saddle']['roll_proxy']:.2f} (conf {metrics['symmetry']['saddle']['confidence']:.1f}%)",
        "HORSE_TOPLINE_STD": f"{metrics['symmetry']['horse']['topline_std']:.2f}",
        "CONF_NOTE": metrics['symmetry']['horse']['note'],
        "GEAR_USED_BLOCK": gear_used_block,
        "GEAR_NOTES_BLOCK": gear_notes_block,
        "GEAR_DETECTED_BLOCK": gear_detected_block,
        "DETECT_NOTES_BLOCK": detect_notes_block,
        "GEAR_RECS_BLOCK": gear_recs_block,
        "MARKS_ROWS": marks_rows,
        "STICK_SVG": stick_svg,
        "GROWTH_SVG": growth_svg,
        "MARK_CHART_SVG": mark_chart_svg,
        "COMPARISON_BLOCK": comparison_block,
        "ANALYSIS_SECTIONS": analysis_sections_html,
        "PDF_ITEM": pdf_item,
        "REINITS": int(metrics.get('tracking', {}).get('reinitializations', 0)),
        "GEAR_HINT": "auto-detected" if len(detected_gear) > 0 else "check video clarity",
        "REC_JS": rec_js,
        "SERIES_JS": series_js,
    }

    html_built = template
    for key, val in replacements.items():
        html_built = html_built.replace(f"%%{key}%%", str(val))

    return html_built




def render_pdf_report_html(
    analysis_id: str,
    meta: dict,
    metrics: Dict[str, Any],
    scored: Dict,
    mark_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    analysis_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Rich PDF layout for the downloadable report.
    """
    mark_scores = mark_scores or {}
    series = metrics.get("series", {}) or {}
    flags = scored.get("flags", []) or []
    rec_list = scored.get("recommendations", []) or []
    coach = scored.get("coach", {}) or {}
    doing_well = coach.get("doing_well", []) or []
    to_improve = coach.get("to_improve", []) or []
    drills = coach.get("drills", []) or []
    gear_assessment = scored.get("gear_assessment", {}) or {}
    gear_detection = scored.get("gear_detection", {}) or meta.get("gear_detection", {}) or {}
    detected_gear = gear_detection.get("gear", {}) or {}
    gear_conf = gear_detection.get("confidences", {}) or {}
    gear_used = scored.get("gear_used", {}) or meta.get("gear_used", {}) or meta.get("gear", {}) or {}

    def tr(label: str, val: str) -> str:
        return f"<tr><td>{label}</td><td>{val}</td></tr>"

    used_rows = ""
    for key in ["helmet", "boots", "pad", "girth", "breastplate", "crupper", "stirrups"]:
        used_rows += tr(f"Used {key.title()}", gear_used.get(key, "-"))
    detected_rows = ""
    for key, val in detected_gear.items():
        detected_rows += tr(f"Detected {key.title()}", f"{val} ({int(gear_conf.get(key, 0.0) * 100)}%)")
    detect_notes = " / ".join(gear_detection.get("notes", []) or ["-"])
    gear_used_summary = ", ".join([f"{k}:{gear_used.get(k,'')}" for k in gear_used]) if gear_used else "-"
    gear_detected_summary = ", ".join([f"{k}:{v} ({int(gear_conf.get(k,0.0)*100)}%)" for k, v in detected_gear.items()]) if detected_gear else "-"

    mark_rows = ""
    for name, data in mark_scores.items():
        mark_rows += tr(f"{name} (10)", f"{data.get('value',0):.1f} - {data.get('improve','')}")

    flags_html = "<p class='muted'>No warnings flagged.</p>" if len(flags) == 0 else "<ul>" + "".join([f"<li>{x}</li>" for x in flags]) + "</ul>"
    rec_html = "<p class='muted'>No recommendations generated.</p>" if len(rec_list) == 0 else "<ul>" + "".join([f"<li>{x}</li>" for x in rec_list]) + "</ul>"
    doing_html = "<p class='muted'>No positives detected.</p>" if len(doing_well) == 0 else "<ul>" + "".join([f"<li>{x}</li>" for x in doing_well]) + "</ul>"
    improve_html = "<p class='muted'>No improvement items.</p>" if len(to_improve) == 0 else "<ul>" + "".join([f"<li>{x}</li>" for x in to_improve]) + "</ul>"
    drills_html = "<p class='muted'>No drills listed.</p>" if len(drills) == 0 else "<ol>" + "".join([f"<li>{x}</li>" for x in drills]) + "</ol>"

    warnings_txt = "; ".join(flags) if flags else "None noted."
    top_tip = rec_list[0] if len(rec_list) > 0 else "Keep your current balance and rhythm steady."
    tracking_pct = metrics.get("tracking", {}).get("tracking_success_pct", 0.0)
    frames_seen = int(metrics.get("tracking", {}).get("frames", metrics.get("frames_analyzed", 0)))
    stick_svg = build_stick_svg(metrics)
    growth_svg = build_growth_svg(metrics, scored["scores"])
    mark_chart_svg = build_mark_chart(mark_scores)
    trend_svg = build_pdf_trend_svg(series)
    if analysis_payload is None:
        analysis_payload = build_analysis_payload(
            analysis_id,
            meta,
            metrics,
            scored,
            mark_scores=mark_scores,
        )
    summary_tiles_html = _score_tiles(analysis_payload.get("summary_cards", scored.get("scores", {})))
    analysis_sections_html = analysis_payload.get("analysis_sections_html", "")
    horse_profile_label = analysis_payload.get("horse_profile_label", meta.get("horse_profile") or meta.get("horse_scheme", ""))
    discipline_label = analysis_payload.get("discipline_label", meta.get("discipline", "general_riding"))
    confidence_pct = float(analysis_payload.get("confidence", 0.0) or 0.0)

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        body {{ font-family: "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 24px; color: #0f172a; background: #f7f8fb; }}
        .wrap {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ margin: 0 0 12px 0; font-size: 28px; }}
        h2 {{ margin: 0 0 10px 0; font-size: 20px; }}
        h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
        p {{ margin: 0 0 8px 0; }}
        ul, ol {{ margin: 6px 0 0 18px; }}
        .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px 16px; box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06); margin-top: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
        .pill {{ display: inline-block; padding: 6px 12px; border-radius: 999px; background: #0ea5e91a; color: #0f172a; border: 1px solid #0ea5e94d; font-size: 12px; }}
        .muted {{ color: #64748b; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 6px; }}
        td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 4px; font-size: 13px; }}
        td:first-child {{ color: #475569; font-weight: 600; width: 60%; }}
        td:last-child {{ text-align: right; color: #0f172a; font-weight: 700; }}
        table tr:last-child td {{ border-bottom: none; }}
        table tr:nth-child(even) td {{ background: #f8fafc; }}
        .label {{ color: #475569; font-size: 13px; }}
        .value {{ font-weight: 700; font-size: 20px; }}
        .section-title {{ margin-bottom: 6px; font-weight: 700; color: #0f172a; }}
        .flex {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
        .viz {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 8px; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="flex" style="justify-content: space-between; align-items: baseline;">
          <div>
            <h1>Riders Bay SaddleFit Report</h1>
            <div class="muted">ID: {analysis_id} | Profile: {horse_profile_label} | Discipline: {discipline_label} | Saddle: {meta.get("saddle_type","")}</div>
            <div class="muted">Analysis confidence: {confidence_pct:.1f}%</div>
          </div>
          <div class="pill">Professional Summary</div>
        </div>

        <div class="summary">
          <div class="card">
            <div class="label">Stability</div>
            <div class="value">{scored["scores"]["saddle_stability"]}/100</div>
            <div class="muted">{scored["scores"]["stability_label"]}</div>
          </div>
          <div class="card">
            <div class="label">Fit risk</div>
            <div class="value">{scored["scores"]["fit_risk"]}</div>
          </div>
          <div class="card">
            <div class="label">Rider level</div>
            <div class="value">{scored["scores"]["rider_level"]}</div>
            <div class="muted">Score {scored["scores"]["rider_score"]}/100</div>
          </div>
          <div class="card">
            <div class="label">Tracking quality</div>
            <div class="value">{tracking_pct:.1f}%</div>
            <div class="muted">{frames_seen} frames analyzed</div>
          </div>
          <div class="card">
            <div class="label">Gear</div>
            <div class="value">{gear_assessment.get("status", "PASS")}</div>
            <div class="muted">{gear_used_summary}</div>
          </div>
        </div>

        <div class="card">
          <h2>Score Summary</h2>
          <div class="summary">
            {summary_tiles_html}
          </div>
        </div>

        <div class="card">
          <h2>Quick Summary</h2>
          <ul>
            <li>Overall stability: <b>{scored["scores"]["stability_label"]}</b> ({scored["scores"]["saddle_stability"]}/100).</li>
            <li>Fit risk: <b>{scored["scores"]["fit_risk"]}</b>; rider level: <b>{scored["scores"]["rider_level"]}</b> ({scored["scores"]["rider_score"]}/100).</li>
            <li>Warnings: {warnings_txt}</li>
            <li>Video quality: {tracking_pct:.1f}% tracking across {frames_seen} frames.</li>
            <li>Next step: {top_tip}</li>
          </ul>
        </div>

        <div class="card grid">
          <div>
            <h3 class="section-title">Flags</h3>
            {flags_html}
          </div>
          <div>
            <h3 class="section-title">Recommendations</h3>
            {rec_html}
          </div>
        </div>

        <div class="card">
          <h2>Visual Snapshot</h2>
          <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));">
            <div class="viz">{stick_svg}</div>
            <div class="viz">{growth_svg}</div>
            <div class="viz">{mark_chart_svg}</div>
            <div class="viz">{trend_svg}</div>
          </div>
        </div>

        <div class="card grid">
          <div>
            <h3 class="section-title">Saddle Stability Metrics</h3>
            <table>
              {tr("Pitch mean / std (deg)", f"{metrics['pitch_mean_deg']:.2f} / {metrics['pitch_std_deg']:.2f}")}
              {tr("Rocking amplitude (deg)", f"{metrics['rock_amplitude_deg']:.2f}")}
              {tr("Drift X (px) + direction", f"{metrics['mid_drift_x_px']:.2f} ({metrics.get('mid_drift_direction','stable')})")}
              {tr("Drift rate (px/s)", f"{metrics.get('mid_drift_rate_px_s',0.0):.2f}")}
              {tr("Vertical bounce (px)", f"{metrics.get('mid_bounce_y_px',0.0):.2f}")}
              {tr("Cadence (Hz)", f"{metrics['cadence_hz']:.2f}")}
              {tr("Duration (sec)", f"{metrics.get('duration_sec',0.0):.2f}")}
              {tr("Frames analyzed", f"{int(metrics.get('frames_analyzed',0))}")}
            </table>
          </div>
          <div>
            <h3 class="section-title">Alignment</h3>
            <table>
              {tr("Topline mean / std (deg)", f"{metrics['alignment']['topline_mean_deg']:.2f} / {metrics['alignment']['topline_std_deg']:.2f}")}
              {tr("Saddle vs topline diff (deg)", f"{metrics['alignment']['saddle_topline_diff_deg']:.2f}")}
              {tr("Withers clearance min/mean/std (px)", f"{metrics['alignment']['withers_clearance_min_px']:.2f} / {metrics['alignment']['withers_clearance_mean_px']:.2f} / {metrics['alignment']['withers_clearance_std_px']:.2f}")}
              {tr("Clearance collapse", "Yes" if metrics['alignment']['clearance_collapse'] else "No")}
              {tr("Bridging proxy", f"{metrics['alignment']['bridging_proxy']:.2f}")}
            </table>
          </div>
        </div>

        <div class="card grid">
          <div>
            <h3 class="section-title">Symmetry (rider / saddle / horse)</h3>
            <table>
              {tr("Shoulder diff mean / std (px)", f"{metrics['symmetry']['rider']['shoulder_diff_mean']:.2f} / {metrics['symmetry']['rider']['shoulder_diff_std']:.2f}")}
              {tr("Hip diff mean / std (px)", f"{metrics['symmetry']['rider']['hip_diff_mean']:.2f} / {metrics['symmetry']['rider']['hip_diff_std']:.2f}")}
              {tr("Lean consistency (std deg)", f"{metrics['symmetry']['rider']['lean_consistency_std']:.2f}")}
              {tr("Saddle roll proxy", f"{metrics['symmetry']['saddle']['roll_proxy']:.2f} (conf {metrics['symmetry']['saddle']['confidence']:.1f}%)")}
              {tr("Horse topline std (deg)", f"{metrics['symmetry']['horse']['topline_std']:.2f}")}
              {tr("Confidence note", metrics['symmetry']['horse']['note'])}
            </table>
          </div>
          <div>
            <h3 class="section-title">Gear & Safety</h3>
            <table>
              {tr("Status", gear_assessment.get("status", "PASS"))}
              {tr("Used gear", gear_used_summary)}
              {tr("Detected gear", gear_detected_summary)}
              {tr("Detection notes", detect_notes)}
              {tr("Notes", " / ".join(gear_assessment.get("notes", []) or ["-"]))}
              {tr("Recommendations", " / ".join(gear_assessment.get("recommendations", []) or ["-"]))}
            </table>
          </div>
        </div>

        <div class="card grid">
          <div>
            <h3 class="section-title">Marks (0-10) & How to Improve</h3>
            <table>
              {mark_rows}
            </table>
          </div>
          <div>
            <h3 class="section-title">Coaching</h3>
            <div class="label" style="margin-bottom:4px;">Doing well</div>
            {doing_html}
            <div class="label" style="margin-top:8px;">To improve</div>
            {improve_html}
            <div class="label" style="margin-top:8px;">Drills</div>
            {drills_html}
          </div>
        </div>

        <div class="card">
          <h3 class="section-title">Tracking Confidence & Video Quality</h3>
          <table>
            {tr("Tracking success (%)", f"{metrics.get('tracking', {}).get('tracking_success_pct',0.0):.1f}%")}
            {tr("Frames processed", f"{int(metrics.get('tracking', {}).get('frames', metrics.get('frames_analyzed',0)))}")}
            {tr("Reinitializations", f"{int(metrics.get('tracking', {}).get('reinitializations', 0))}")}
            {tr("Confidence note", "Higher % means more reliable symmetry; side-view limits roll estimation.")}
          </table>
        </div>

        <div class="card">
          <h3 class="section-title">Riding Improvement Tips</h3>
          <p class="muted">Use these in your next session.</p>
          {rec_html}
        </div>

        {analysis_sections_html}
      </div>
    </body>
    </html>
    """


def save_report_assets(
    run_dir: str,
    analysis_id: str,
    meta: dict,
    metrics: Dict[str, Any],
    scored: Dict,
    points: Optional[Dict[str, List[float]]],
) -> Tuple[bool, str]:
    storage.ensure_runtime_directories()
    mark_scores = compute_mark_scores(metrics, scored)
    stick_svg = build_stick_svg(metrics, points)
    growth_svg = build_growth_svg(metrics, scored["scores"])
    mark_chart_svg = build_mark_chart(mark_scores)
    analysis_payload = scored.get("analysis_payload") or build_analysis_payload(
        analysis_id,
        meta,
        metrics,
        scored,
        points=points,
        pose_summary=scored.get("pose_summary", {}),
        mark_scores=mark_scores,
    )
    scored["analysis_payload"] = analysis_payload
    report_dir = storage.analysis_report_dir(analysis_id)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_html = render_report_html(
        analysis_id,
        meta,
        metrics,
        scored,
        stick_svg,
        growth_svg,
        mark_chart_svg,
        pdf_available=True,  # optimistic for PDF generation run
        analysis_payload=analysis_payload,
        points=points,
        mark_scores=mark_scores,
    )

    pdf_path = report_dir / "report.pdf"
    pdf_generated = ensure_pdf_file(analysis_id, meta, metrics, scored, mark_scores, pdf_path)

    final_html = render_report_html(
        analysis_id,
        meta,
        metrics,
        scored,
        stick_svg,
        growth_svg,
        mark_chart_svg,
        pdf_available=pdf_generated,
        analysis_payload=analysis_payload,
        points=points,
        mark_scores=mark_scores,
    )
    with open(report_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(final_html)

    return pdf_generated, final_html


def render_comparison_block(label_a: str, label_b: str, metrics_a: Dict[str, float], metrics_b: Dict[str, float], scores_a: Dict[str, float], scores_b: Dict[str, float]) -> str:
    rows = [
        ("Saddle Stability", scores_a.get("saddle_stability", 0.0), scores_b.get("saddle_stability", 0.0)),
        ("Rider Score", scores_a.get("rider_score", 0.0), scores_b.get("rider_score", 0.0)),
        ("Pitch mean (deg)", metrics_a["pitch_mean_deg"], metrics_b["pitch_mean_deg"]),
        ("Rock amplitude (deg)", metrics_a["rock_amplitude_deg"], metrics_b["rock_amplitude_deg"]),
        ("Mid drift X (px)", metrics_a["mid_drift_x_px"], metrics_b["mid_drift_x_px"]),
        ("Cadence (Hz)", metrics_a["cadence_hz"], metrics_b["cadence_hz"]),
    ]

    row_html = ""
    for name, va, vb in rows:
        delta = vb - va
        delta_txt = f"{delta:+.2f}"
        row_html += f"<tr><td>{name}</td><td>{va:.2f}</td><td>{vb:.2f}</td><td>{delta_txt}</td></tr>"

    return f"""
    <div class="card">
      <h3>Comparison</h3>
      <table>
        <tr><th>Metric</th><th>{label_a}</th><th>{label_b}</th><th>Delta (B-A)</th></tr>
        {row_html}
      </table>
    </div>
    """


def render_compare_report(
    compare_id: str,
    meta: dict,
    label_a: str,
    label_b: str,
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    scored_a: Dict,
    scored_b: Dict,
    pdf_available: bool,
    comparison_payload: Optional[Dict[str, Any]] = None,
    show_video: bool = True,
    include_analysis: bool = False,
    mark_scores_a: Optional[Dict[str, Dict[str, Any]]] = None,
    mark_scores_b: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    mark_scores_a = mark_scores_a or compute_mark_scores(metrics_a, scored_a)
    mark_scores_b = mark_scores_b or compute_mark_scores(metrics_b, scored_b)
    comparison_payload = comparison_payload or {}
    comparison_sections_html = comparison_payload.get("comparison_sections_html", "")
    comparison_summary = comparison_payload.get("overall_summary", "")

    def list_html(items, empty_msg: str) -> str:
        if not items:
            return f"<p class='muted'>{empty_msg}</p>"
        return "<ul>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"

    def gear_summary(scored: Dict[str, Any]) -> Tuple[str, str, str, str]:
        gear_assessment = scored.get("gear_assessment", {}) or {}
        gear_detection = scored.get("gear_detection", {}) or {}
        gear_used = scored.get("gear_used", {}) or {}
        gear_conf = gear_detection.get("confidences", {}) or {}
        used_txt = ", ".join([f"{k}:{v}" for k, v in gear_used.items()]) if gear_used else "-"
        detected_list = [f"{k}:{v} ({int(gear_conf.get(k, 0.0) * 100)}%)" for k, v in (gear_detection.get("gear", {}) or {}).items()]
        detected_txt = ", ".join(detected_list) if detected_list else "-"
        detect_notes = " / ".join(gear_detection.get("notes", []) or ["-"])
        recs = " / ".join(gear_assessment.get("recommendations", []) or ["-"])
        return used_txt, detected_txt, detect_notes, recs

    def summary_card(label: str, metrics: Dict[str, Any], scored: Dict[str, Any]) -> str:
        track = metrics.get("tracking", {}) or {}
        return f"""
        <div class="stat">
          <div class="k">{label}</div>
          <div class="v">{scored['scores'].get('saddle_stability',0)}/100</div>
          <div class="muted">Rider {scored['scores'].get('rider_level','')} ({scored['scores'].get('rider_score',0)}/100)</div>
          <div class="muted">Fit risk: {scored['scores'].get('fit_risk','')}</div>
          <div class="muted">Tracking {track.get('tracking_success_pct',0.0):.1f}% · {int(track.get('frames', metrics.get('frames_analyzed',0)))} frames</div>
        </div>
        """

    def detail_block(label: str, metrics: Dict[str, Any], scored: Dict[str, Any], marks: Dict[str, Any]) -> str:
        flags_block = list_html(scored.get("flags", []), "No issues flagged.")
        recs_block = list_html(scored.get("recommendations", []), "No recommendations generated.")
        doing_block = list_html(scored.get("coach", {}).get("doing_well", []), "No positives detected.")
        improve_block = list_html(scored.get("coach", {}).get("to_improve", []), "No improvement items.")
        drills_block = list_html(scored.get("coach", {}).get("drills", []), "No drills provided.")
        used_txt, detected_txt, detect_notes, recs = gear_summary(scored)
        align = metrics.get("alignment", {}) or {}
        sym_rider = metrics.get("symmetry", {}).get("rider", {}) if metrics.get("symmetry") else {}
        sym_saddle = metrics.get("symmetry", {}).get("saddle", {}) if metrics.get("symmetry") else {}
        sym_horse = metrics.get("symmetry", {}).get("horse", {}) if metrics.get("symmetry") else {}
        mark_rows = "".join([f"<tr><td>{name}</td><td>{data.get('value',0):.1f}/10</td><td>{data.get('improve','')}</td></tr>" for name, data in marks.items()]) or "<tr><td colspan='3'>No marks available.</td></tr>"
        tracking = metrics.get("tracking", {}) or {}

        return f"""
        <div class="card inner">
          <div class="section-title">{label}</div>
          <div class="muted" style="margin-bottom:8px;">Stability {scored['scores'].get('stability_label','')} · Fit risk {scored['scores'].get('fit_risk','')} · Rider {scored['scores'].get('rider_level','')} ({scored['scores'].get('rider_score',0)}/100)</div>
          <div class="subcard">
            <div class="section-title">Flags & recommendations</div>
            <div class="k">Flags</div>{flags_block}
            <div class="k" style="margin-top:8px;">Recommendations</div>{recs_block}
          </div>
          <div class="subcard">
            <div class="section-title">Key metrics</div>
            <table>
              <tr><td>Pitch mean / std (deg)</td><td>{metrics.get('pitch_mean_deg',0.0):.2f} / {metrics.get('pitch_std_deg',0.0):.2f}</td></tr>
              <tr><td>Rock amplitude (deg)</td><td>{metrics.get('rock_amplitude_deg',0.0):.2f}</td></tr>
              <tr><td>Drift X (px + direction)</td><td>{metrics.get('mid_drift_x_px',0.0):.2f} ({metrics.get('mid_drift_direction','stable')})</td></tr>
              <tr><td>Drift rate (px/s)</td><td>{metrics.get('mid_drift_rate_px_s',0.0):.2f}</td></tr>
              <tr><td>Vertical bounce (px)</td><td>{metrics.get('mid_bounce_y_px',0.0):.2f}</td></tr>
              <tr><td>Cadence (Hz)</td><td>{metrics.get('cadence_hz',0.0):.2f}</td></tr>
              <tr><td>Duration (sec)</td><td>{metrics.get('duration_sec',0.0):.2f}</td></tr>
              <tr><td>Frames analyzed</td><td>{int(metrics.get('frames_analyzed',0))}</td></tr>
              <tr><td>Tracking success (%)</td><td>{tracking.get('tracking_success_pct',0.0):.1f}% · {int(tracking.get('frames', metrics.get('frames_analyzed',0)))} frames</td></tr>
            </table>
            <div class="section-title" style="margin-top:10px;">Alignment & symmetry</div>
            <table>
              <tr><td>Topline mean / std (deg)</td><td>{align.get('topline_mean_deg',0.0):.2f} / {align.get('topline_std_deg',0.0):.2f}</td></tr>
              <tr><td>Saddle vs topline diff (deg)</td><td>{align.get('saddle_topline_diff_deg',0.0):.2f}</td></tr>
              <tr><td>Withers clearance min/mean/std (px)</td><td>{align.get('withers_clearance_min_px',0.0):.2f} / {align.get('withers_clearance_mean_px',0.0):.2f} / {align.get('withers_clearance_std_px',0.0):.2f}</td></tr>
              <tr><td>Clearance collapse</td><td>{"Yes" if align.get('clearance_collapse') else "No"}</td></tr>
              <tr><td>Bridging proxy</td><td>{align.get('bridging_proxy',0.0):.2f}</td></tr>
              <tr><td>Shoulder diff mean/std (px)</td><td>{sym_rider.get('shoulder_diff_mean',0.0):.2f} / {sym_rider.get('shoulder_diff_std',0.0):.2f}</td></tr>
              <tr><td>Hip diff mean/std (px)</td><td>{sym_rider.get('hip_diff_mean',0.0):.2f} / {sym_rider.get('hip_diff_std',0.0):.2f}</td></tr>
              <tr><td>Saddle roll proxy</td><td>{sym_saddle.get('roll_proxy',0.0):.2f} (conf {sym_saddle.get('confidence',0.0):.1f}%)</td></tr>
              <tr><td>Horse topline std (deg)</td><td>{sym_horse.get('topline_std',0.0):.2f} ({sym_horse.get('note','')})</td></tr>
            </table>
          </div>
          <div class="subcard">
            <div class="section-title">Gear & safety</div>
            <table>
              <tr><td>Status</td><td>{scored.get('gear_assessment', {}).get('status','PASS')}</td></tr>
              <tr><td>Used</td><td>{used_txt}</td></tr>
              <tr><td>Detected</td><td>{detected_txt}</td></tr>
              <tr><td>Detection notes</td><td>{detect_notes}</td></tr>
              <tr><td>Recommendations</td><td>{recs}</td></tr>
            </table>
          </div>
          <div class="subcard">
            <div class="section-title">Marks & coaching</div>
            <table>{mark_rows}</table>
            <div class="k" style="margin-top:10px;">Doing well</div>{doing_block}
            <div class="k" style="margin-top:6px;">To improve</div>{improve_block}
            <div class="k" style="margin-top:6px;">Drills</div>{drills_block}
          </div>
        </div>
        """

    if include_analysis:
        comp_rows = [
            ("Saddle Stability", scored_a["scores"].get("saddle_stability", 0.0), scored_b["scores"].get("saddle_stability", 0.0)),
            ("Rider Score", scored_a["scores"].get("rider_score", 0.0), scored_b["scores"].get("rider_score", 0.0)),
            ("Pitch mean (deg)", metrics_a.get("pitch_mean_deg", 0.0), metrics_b.get("pitch_mean_deg", 0.0)),
            ("Rock amplitude (deg)", metrics_a.get("rock_amplitude_deg", 0.0), metrics_b.get("rock_amplitude_deg", 0.0)),
            ("Mid drift X (px)", metrics_a.get("mid_drift_x_px", 0.0), metrics_b.get("mid_drift_x_px", 0.0)),
            ("Cadence (Hz)", metrics_a.get("cadence_hz", 0.0), metrics_b.get("cadence_hz", 0.0)),
        ]
        comp_rows_html = ""
        for name, va, vb in comp_rows:
            delta = vb - va
            comp_rows_html += f"<tr><td>{name}</td><td>{va:.2f}</td><td>{vb:.2f}</td><td>{delta:+.2f}</td></tr>"
        comparison_table = f"<table><tr><th>Metric</th><th>{label_a}</th><th>{label_b}</th><th>Delta (B-A)</th></tr>{comp_rows_html}</table>"
        return f"""
        <html>
        <head>
          <title>Comparison Report - {compare_id}</title>
          <meta charset="utf-8" />
          <style>
            body {{ font-family: "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 24px; background:#f6f8fb; color:#0f172a; }}
            .wrap {{ max-width: 1180px; margin: 0 auto; }}
            .header {{ display:flex; justify-content: space-between; align-items: baseline; gap: 12px; }}
            .pill {{ display:inline-block; padding: 6px 12px; border-radius: 999px; background:#ecfdf3; border:1px solid #bbf7d0; color:#166534; font-weight:700; font-size:12px; }}
            .muted {{ color:#475569; font-size:12px; }}
            .section-title {{ font-weight:800; margin:0 0 6px 0; color:#0f172a; }}
            .card {{ background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; box-shadow:0 8px 22px rgba(15,23,42,0.08); padding:14px 16px; }}
            .inner {{ margin-top:12px; }}
            .subcard {{ border:1px solid #e2e8f0; border-radius:12px; padding:10px 12px; margin-top:10px; background:#f8fafc; }}
            .grid {{ display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
            .stat {{ background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:12px 14px; box-shadow:0 4px 14px rgba(15,23,42,0.05); }}
            .k {{ color:#475569; font-size:13px; }}
            .v {{ font-size:22px; font-weight:800; }}
            table {{ width:100%; border-collapse: collapse; margin-top:6px; }}
            td {{ padding:8px 6px; border-bottom:1px solid #e2e8f0; font-size:13px; color:#0f172a; }}
            td:first-child {{ color:#475569; width:60%; }}
            table tr:last-child td {{ border-bottom:none; }}
            ul {{ margin:6px 0 0 18px; }}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="header">
              <div>
                <h1 style="margin:0;">Comparison Report</h1>
                <div class="muted">Side-by-side saddle stability, rider balance, and gear checks.</div>
                <div class="muted" style="margin-top:4px;">{html_escape(str(comparison_summary))}</div>
              </div>
              <div>
                <span class="pill">ID: {compare_id}</span>
                <span class="pill">Profile: {meta.get("horse_profile", meta.get("horse_scheme",""))}</span>
                <span class="pill">Discipline: {meta.get("discipline","general_riding")}</span>
                <span class="pill">Saddle: {meta.get("saddle_type","")}</span>
              </div>
            </div>

            <div class="grid" style="margin-top:14px;">
              {summary_card(label_a, metrics_a, scored_a)}
              {summary_card(label_b, metrics_b, scored_b)}
            </div>

            <div class="grid" style="margin-top:14px;">
              {detail_block(label_a, metrics_a, scored_a, mark_scores_a)}
              {detail_block(label_b, metrics_b, scored_b, mark_scores_b)}
            </div>

            <div class="card" style="margin-top:14px;">
              <div class="section-title">Side-by-side metrics</div>
              {comparison_table}
            </div>

            {comparison_sections_html}
          </div>
        </body>
        </html>
        """

    stick_a = build_stick_svg(metrics_a)
    stick_b = build_stick_svg(metrics_b)
    growth_a = build_growth_svg(metrics_a, scored_a["scores"])
    growth_b = build_growth_svg(metrics_b, scored_b["scores"])
    comparison_block = render_comparison_block(label_a, label_b, metrics_a, metrics_b, scored_a["scores"], scored_b["scores"])
    rec_a = " ".join(scored_a.get("recommendations", [])) if scored_a.get("recommendations") else "Maintain balanced posture and steady cadence."
    rec_b = " ".join(scored_b.get("recommendations", [])) if scored_b.get("recommendations") else "Maintain balanced posture and steady cadence."
    rec_a_js = json.dumps(rec_a)
    rec_b_js = json.dumps(rec_b)

    pdf_item = (
        "<li><a href='/compare_report/" + compare_id + ".pdf' target='_blank'>Download PDF</a></li>"
        if pdf_available
        else "<li>PDF not available (install weasyprint and rerun).</li>"
    )

    video_block = ""
    if show_video:
        video_block = f"""
        <div class="card">
          <h3>Voice Instructor</h3>
          <div class="grid">
            <div>
              <div class="k" style="margin-bottom:6px;">{label_a} video</div>
              <video id="videoA" controls style="width:100%; border-radius:12px; border:1px solid rgba(255,255,255,0.12);">
                <source src="/compare_video/{compare_id}/a" type="video/mp4">
              </video>
              <button style="margin-top:10px; width:100%; background: linear-gradient(135deg,#60a5fa,#a78bfa);" onclick="playCoachA()">Play video + voice tips (A)</button>
            </div>
            <div>
              <div class="k" style="margin-bottom:6px;">{label_b} video</div>
              <video id="videoB" controls style="width:100%; border-radius:12px; border:1px solid rgba(255,255,255,0.12);">
                <source src="/compare_video/{compare_id}/b" type="video/mp4">
              </video>
              <button style="margin-top:10px; width:100%; background: linear-gradient(135deg,#60a5fa,#a78bfa);" onclick="playCoachB()">Play video + voice tips (B)</button>
            </div>
          </div>
          <script>
            const recTextA = {rec_a_js};
            const recTextB = {rec_b_js};
            function playCoachA() {{
              const vid = document.getElementById("videoA");
              if (vid) {{ try {{ vid.currentTime = 0; vid.play(); }} catch(e) {{}} }}
              if ('speechSynthesis' in window) {{
                const u = new SpeechSynthesisUtterance("Coach tips: " + recTextA);
                u.rate = 1.0;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
              }}
            }}
            function playCoachB() {{
              const vid = document.getElementById("videoB");
              if (vid) {{ try {{ vid.currentTime = 0; vid.play(); }} catch(e) {{}} }}
              if ('speechSynthesis' in window) {{
                const u = new SpeechSynthesisUtterance("Coach tips: " + recTextB);
                u.rate = 1.0;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
              }}
            }}
          </script>
        </div>
        """

    return f"""
    <html>
    <head>
      <title>Comparison Report - {compare_id}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        body {{ font-family: Arial; margin:0; background:#030712; color:#e5e7eb; }}
        .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px 18px; }}
        .card {{ padding: 18px; border-radius: 16px; background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.10); margin-top: 14px; box-shadow: 0 16px 40px rgba(0,0,0,0.25); }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
        .k {{ color:#cbd5e1; font-size: 13px; }}
        .v {{ font-size: 22px; font-weight: 800; }}
        .pill {{ display:inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(34,197,94,0.18); border: 1px solid rgba(34,197,94,0.35); }}
        a {{ color:#60a5fa; }}
        table {{ width:100%; border-collapse: collapse; margin-top: 10px; }}
        td, th {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.10); }}
        h2, h3 {{ margin: 0; }}
      </style>
    </head>
    <body>
        <div class="wrap">
        <h2 style="margin:0;">Video Comparison Report</h2>
        <div style="margin-top:8px;">
          <span class="pill">ID: {compare_id}</span>
          <span class="pill">Profile: {meta.get("horse_profile", meta.get("horse_scheme",""))}</span>
          <span class="pill">Discipline: {meta.get("discipline","general_riding")}</span>
          <span class="pill">Saddle: {meta.get("saddle_type","")}</span>
        </div>

        <div class="card grid">
          <div>
            <div class="k">{label_a} - Saddle Stability</div>
            <div class="v">{scored_a["scores"]["saddle_stability"]}/100</div>
          </div>
          <div>
            <div class="k">{label_b} - Saddle Stability</div>
            <div class="v">{scored_b["scores"]["saddle_stability"]}/100</div>
          </div>
          <div>
            <div class="k">{label_a} - Rider Level</div>
            <div class="v">{scored_a["scores"]["rider_level"]} ({scored_a["scores"]["rider_score"]}/100)</div>
          </div>
          <div>
            <div class="k">{label_b} - Rider Level</div>
            <div class="v">{scored_b["scores"]["rider_level"]} ({scored_b["scores"]["rider_score"]}/100)</div>
          </div>
        </div>

        {comparison_block}

        {comparison_sections_html}

        <div class="card">
          <h3>Visual Aids</h3>
          <div class="grid">
            <div>
              <div class="k" style="margin-bottom:6px;">{label_a} posture</div>
              {stick_a}
              {growth_a}
            </div>
            <div>
              <div class="k" style="margin-bottom:6px;">{label_b} posture</div>
              {stick_b}
              {growth_b}
            </div>
          </div>
        </div>

        {video_block}

        <div class="card">
          <h3>Open Files</h3>
          <ul>
            <li><a href="/compare_report/{compare_id}" target="_blank">This report (HTML)</a></li>
            {pdf_item}
            <li><a href="/">Analyze another video</a></li>
          </ul>
        </div>
      </div>
    </body>
    </html>
    """


def save_compare_report(
    compare_dir: str,
    compare_id: str,
    meta: dict,
    label_a: str,
    label_b: str,
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    scored_a: Dict,
    scored_b: Dict,
    comparison_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    storage.ensure_runtime_directories()
    mark_scores_a = compute_mark_scores(metrics_a, scored_a)
    mark_scores_b = compute_mark_scores(metrics_b, scored_b)
    report_dir = storage.comparison_report_dir(compare_id)
    report_dir.mkdir(parents=True, exist_ok=True)

    if comparison_payload is None:
        comparison_payload = build_comparison_payload(compare_id, meta, label_a, label_b, scored_a.get("analysis_payload", {}), scored_b.get("analysis_payload", {}))

    pdf_html = render_compare_report(
        compare_id,
        meta,
        label_a,
        label_b,
        metrics_a,
        metrics_b,
        scored_a,
        scored_b,
        pdf_available=True,
        comparison_payload=comparison_payload,
        show_video=False,
        include_analysis=True,
        mark_scores_a=mark_scores_a,
        mark_scores_b=mark_scores_b,
    )

    pdf_path = report_dir / "report.pdf"
    pdf_generated = False
    pdf_renderer = get_weasyprint()
    if pdf_renderer is not None:
        try:
            pdf_renderer.HTML(string=pdf_html, base_url=BASE_DIR).write_pdf(os.fspath(pdf_path))
            pdf_generated = True
        except Exception:
            pdf_generated = False

    if not pdf_generated:
        rec_a = scored_a.get("recommendations", []) or []
        rec_b = scored_b.get("recommendations", []) or []
        lines = [
            f"Comparison ID: {compare_id}",
            f"Scheme: {meta.get('horse_scheme', '')} | Saddle: {meta.get('saddle_type', '')}",
            f"{label_a} stability: {scored_a['scores'].get('saddle_stability', 0)} ({scored_a['scores'].get('stability_label', '')})",
            f"{label_b} stability: {scored_b['scores'].get('saddle_stability', 0)} ({scored_b['scores'].get('stability_label', '')})",
            f"{label_a} rider score: {scored_a['scores'].get('rider_score', 0)}/100",
            f"{label_b} rider score: {scored_b['scores'].get('rider_score', 0)}/100",
            f"Pitch mean/std: {metrics_a.get('pitch_mean_deg', 0.0):.2f}/{metrics_a.get('pitch_std_deg', 0.0):.2f} vs {metrics_b.get('pitch_mean_deg', 0.0):.2f}/{metrics_b.get('pitch_std_deg', 0.0):.2f}",
            f"Rock amplitude: {metrics_a.get('rock_amplitude_deg', 0.0):.2f} vs {metrics_b.get('rock_amplitude_deg', 0.0):.2f}",
            f"Drift X: {metrics_a.get('mid_drift_x_px', 0.0):.2f} vs {metrics_b.get('mid_drift_x_px', 0.0):.2f}",
            f"Cadence: {metrics_a.get('cadence_hz', 0.0):.2f} vs {metrics_b.get('cadence_hz', 0.0):.2f}",
            f"Flags {label_a}: " + ("; ".join(scored_a.get("flags", []) or ["None"]) if scored_a.get("flags") else "None"),
            f"Flags {label_b}: " + ("; ".join(scored_b.get("flags", []) or ["None"]) if scored_b.get("flags") else "None"),
            f"Top recommendation {label_a}: {rec_a[0] if rec_a else 'Keep a steady rhythm and balanced posture.'}",
            f"Top recommendation {label_b}: {rec_b[0] if rec_b else 'Keep a steady rhythm and balanced posture.'}",
        ]
        pdf_bytes = build_simple_pdf_bytes("Saddle Fit Comparison (lite)", lines)
        with open(pdf_path, "wb") as pf:
            pf.write(pdf_bytes)
        pdf_generated = True

    final_html = render_compare_report(
        compare_id,
        meta,
        label_a,
        label_b,
        metrics_a,
        metrics_b,
        scored_a,
        scored_b,
        pdf_available=pdf_generated,
        comparison_payload=comparison_payload,
        show_video=True,
        include_analysis=False,
        mark_scores_a=mark_scores_a,
        mark_scores_b=mark_scores_b,
    )
    with open(report_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(final_html)

    return pdf_generated, final_html


def analyze_video_auto(run_dir: str, analysis_id: str, meta: dict) -> Tuple[dict, Dict[str, float], Dict, float]:
    run_dir_path = Path(run_dir)
    video_path = run_dir_path / meta["video_filename"]
    frame_path = run_dir_path / "frame0.png"
    frame = cv2.imread(os.fspath(frame_path))
    if frame is None:
        raise RuntimeError("First frame missing for auto-calibration.")

    auto = auto_calibrate_points(frame, meta["horse_scheme"], meta["saddle_type"])
    if auto["points"] is None:
        raise RuntimeError("Auto-detection failed. Please upload a clearer side-view video.")

    points = auto["points"]
    # Auto-detect gear/safety from the first frame (use points for pad band crop when available).
    detection_data = auto_detect_gear_and_safety(frame, points)
    meta["gear_detection"] = detection_data

    merged_gear = merge_gear_sources(detection_data.get("gear", {}), meta.get("gear", {}))
    meta["gear"] = merged_gear
    meta["gear_used"] = merged_gear
    try:
        with open(run_dir_path / "meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
    except Exception:
        pass

    frames, times = extract_frames(os.fspath(video_path), target_fps=12)
    tracks, tstats = track_points_lk(frames, points)
    metrics = compute_metrics(tracks, times, tstats)
    gear_assessment = evaluate_gear(merged_gear, metrics, meta["horse_scheme"])
    pose_summary = sample_rider_pose_metrics(frames, times, points)
    scored = score_and_recommend(
        metrics,
        meta.get("horse_profile", meta.get("horse_scheme", "high_wither")),
        meta["saddle_type"],
        meta.get("discipline", "general_riding"),
        pose_summary,
    )
    scored["gear_assessment"] = gear_assessment
    scored["gear_detection"] = detection_data
    scored["gear_used"] = merged_gear
    scored["pose_summary"] = pose_summary
    mark_scores = compute_mark_scores(metrics, scored)

    result = build_analysis_payload(
        analysis_id,
        meta,
        metrics,
        scored,
        points=points,
        pose_summary=pose_summary,
        mark_scores=mark_scores,
    )
    result["gear"] = meta.get("gear", {})
    result["gear_detected"] = detection_data
    result["gear_detection"] = detection_data
    result["gear_used"] = merged_gear
    result["auto_detect_confidence"] = auto["confidence"]
    result["gear_assessment"] = gear_assessment
    result["marks"] = mark_scores
    result["calibration_points"] = points
    result["horse_scheme"] = meta.get("horse_scheme", meta.get("horse_profile", "high_wither"))
    result["video_filename"] = meta["video_filename"]
    scored["analysis_payload"] = result

    with open(run_dir_path / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result, metrics, scored, auto["confidence"]


# ------------------- UI Pages -------------------
@app.get("/dashboard.js")
def dashboard_js():
    js_path = Path(__file__).with_name("dashboard.js")
    if not js_path.exists():
        return Response("console.error('dashboard.js missing');", media_type="application/javascript", status_code=404)
    return Response(js_path.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    dep_warning = ""
    if not MULTIPART_AVAILABLE:
        dep_warning = """
        <div class="alert">
          <div class="alert-title">Dependency missing</div>
          <div class="alert-body">
            <b>python-multipart</b> is not installed. Uploads will not work until it is installed.<br/>
            Run: <code>pip install python-multipart</code> and restart the server.
          </div>
        </div>
        """
    upload_mode = "blob" if storage.IS_VERCEL else "multipart"
    dashboard_config = json.dumps(
        {
            "uploadMode": upload_mode,
            "storageAccess": get_video_storage_access(),
            "maxVideoBytes": get_video_upload_max_bytes(),
            "maxVideoLabel": f"{get_video_upload_max_bytes() // (1024 * 1024)} MB",
            "blobUploadUrl": "/api/blob-upload",
            "analysisApiUrl": "/api/analyze",
            "compareApiUrl": "/api/compare",
            "isVercel": storage.IS_VERCEL,
        }
    )
    return f"""
    <html>
    <head>
      <title>Riders Bay Saddle Fit Agent</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        :root {{
          --bg: #f6f8fb;
          --fg: #0f172a;
          --muted: #4b5563;
          --card: #ffffff;
          --line: #e5e7eb;
          --accent: #5be286;
          --accent-2: #2dd4bf;
          --shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
          --radius: 18px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: 'Poppins','Inter','Segoe UI',system-ui,-apple-system,sans-serif;
          background: radial-gradient(circle at 10% 20%, #e8fff2 0%, #f6f8fb 25%, #eef7ff 50%, #f6f8fb 100%);
          color: var(--fg);
          min-height: 100vh;
          position: relative;
          overflow-x: hidden;
        }}
        body::after {{
          content:"";
          position: absolute;
          inset: -160px auto auto 60%;
          width: 540px;
          height: 540px;
          background: radial-gradient(circle, rgba(45,212,191,0.25) 0%, rgba(91,226,134,0) 60%);
          filter: blur(6px);
          z-index: 0;
        }}
        a {{ color: #0f766e; text-decoration: none; }}
        .nav {{
          position: sticky;
          top: 0;
          z-index: 10;
          backdrop-filter: blur(8px);
          background: rgba(246,248,251,0.88);
          border-bottom: 1px solid rgba(15,23,42,0.05);
        }}
        .nav-inner {{
          max-width: 1180px;
          margin: 0 auto;
          padding: 14px 18px;
          display: flex;
          align-items: center;
          gap: 18px;
          justify-content: space-between;
        }}
        .brand {{
          font-weight: 800;
          letter-spacing: -0.02em;
          font-size: 18px;
          display: flex;
          align-items: center;
          gap: 8px;
          color: var(--fg);
        }}
        .brand-badge {{
          width: 34px; height: 34px; border-radius: 10px;
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          display:flex; align-items:center; justify-content:center;
          color:#0f172a; font-weight: 800;
        }}
        .nav-links {{ display:flex; gap:14px; align-items:center; }}
        .nav-links a {{ color: var(--muted); font-weight: 600; }}
        .nav-cta {{ display:flex; gap:10px; align-items:center; }}
        .btn {{
          display:inline-flex; align-items:center; justify-content:center;
          padding: 10px 14px; border-radius: 999px;
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          color: #0f172a; font-weight: 800; border: none; cursor: pointer;
          box-shadow: 0 12px 28px rgba(45,212,191,0.35);
        }}
        .ghost {{
          padding: 10px 14px; border-radius: 999px; border: 1px solid var(--line);
          background: #ffffff; color: var(--fg); font-weight: 700; cursor: pointer;
        }}
        .container {{ max-width: 1180px; margin: 0 auto; padding: 38px 18px 80px; position: relative; z-index: 1; }}
        .hero {{
          display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 26px; align-items: stretch;
          padding: 28px; border-radius: 24px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow);
          position: relative; overflow: hidden;
        }}
        .hero::before {{
          content:""; position:absolute; inset:-40px auto auto -80px; width: 220px; height: 220px;
          background: radial-gradient(circle, rgba(91,226,134,0.25), rgba(91,226,134,0));
          filter: blur(2px); z-index: 0;
        }}
        .hero::after {{
          content:""; position:absolute; inset:auto -60px -80px auto; width: 280px; height: 280px;
          background: radial-gradient(circle, rgba(45,212,191,0.15), rgba(45,212,191,0));
          filter: blur(2px); z-index: 0;
        }}
        .hero-copy {{ position: relative; z-index: 1; }}
        .eyebrow {{
          display:inline-flex; align-items:center; gap:8px; padding: 6px 12px; border-radius: 999px;
          background: #ecfdf3; color: #166534; font-weight: 700; font-size: 13px; border: 1px solid #bbf7d0;
        }}
        h1 {{ margin: 14px 0 8px 0; font-size: 34px; letter-spacing: -0.02em; }}
        .lede {{ color: var(--muted); line-height: 1.7; margin: 0 0 14px 0; }}
        .hero-tags {{ display:flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
        .tag {{
          padding: 6px 10px; border-radius: 10px; background: #0f172a; color: #ecfeff; font-weight: 600;
          box-shadow: 0 8px 18px rgba(15,23,42,0.18);
        }}
        .actions {{ display:flex; gap:12px; align-items:center; margin-top: 16px; flex-wrap: wrap; }}
        .actions .link {{ color: var(--fg); font-weight: 700; display:flex; align-items:center; gap:8px; }}
        .start-card {{
          position: relative; z-index: 1;
          background: linear-gradient(145deg, #0f172a, #102a43);
          color: #e5e7eb; border-radius: 18px; padding: 20px 20px 22px 20px;
          border: 1px solid rgba(255,255,255,0.06); box-shadow: 0 18px 48px rgba(15,23,42,0.35);
        }}
        .start-card h3 {{ margin: 0 0 8px 0; }}
        .input-row {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
        label {{ display:block; font-weight: 700; margin-top: 12px; letter-spacing: 0.01em; }}
        input, select {{
          width: 100%; margin-top: 6px; padding: 12px 12px; border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06); color: #e5e7eb;
          outline: none; font-weight: 600;
        }}
        input[type="file"] {{ padding: 10px; background: rgba(255,255,255,0.04); }}
        select {{ scrollbar-color: #0b1220 #0f172a; }}
        .hint {{ color: #cbd5e1; font-size: 13px; margin-top: 8px; line-height: 1.5; }}
        .primary {{
          margin-top: 16px; width: 100%; padding: 13px 14px; border-radius: 12px; border: none;
          background: linear-gradient(135deg, #5be286, #2dd4bf);
          color: #0f172a; font-weight: 800; cursor: pointer; letter-spacing: 0.01em;
          box-shadow: 0 12px 28px rgba(45,212,191,0.28);
        }}
        .section {{ margin-top: 32px; }}
        .section-head {{ display:flex; flex-direction:column; gap:6px; max-width: 780px; }}
        h2 {{ margin: 0; font-size: 26px; letter-spacing:-0.01em; }}
        .muted {{ color: var(--muted); line-height: 1.7; }}
        .stat-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 14px; margin-top: 16px; }}
        .stat {{ padding: 16px; border-radius: 14px; background: #ffffff; border: 1px solid var(--line); box-shadow: 0 10px 26px rgba(15,23,42,0.08); }}
        .stat b {{ font-size: 22px; display:block; color: #0f172a; }}
        .feature-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap: 16px; margin-top: 16px; }}
        .feature {{
          background: #ffffff; border-radius: 16px; border: 1px solid var(--line); padding: 16px;
          box-shadow: 0 10px 30px rgba(15,23,42,0.08);
        }}
        .feature h4 {{ margin: 0 0 6px 0; }}
        .flow {{
          display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap: 14px; margin-top: 18px;
        }}
        .step {{
          background: #0f172a; color: #e5e7eb; border-radius: 16px; padding: 16px; border: 1px solid rgba(255,255,255,0.08);
          box-shadow: 0 14px 32px rgba(15,23,42,0.25);
        }}
        .step-num {{ display:inline-flex; width: 34px; height: 34px; align-items:center; justify-content:center; border-radius: 10px; background: rgba(45,212,191,0.16); font-weight: 800; color: #2dd4bf; }}
        .compare {{
          background: linear-gradient(135deg, #0b2545, #0f172a); color: #e5e7eb; border-radius: 20px; padding: 24px;
          display:grid; grid-template-columns: 1fr 1fr; gap: 20px; border: 1px solid rgba(255,255,255,0.06); box-shadow: 0 18px 46px rgba(15,23,42,0.32);
        }}
        .compare h3 {{ margin: 0; }}
        .compare .muted {{ color: #cbd5e1; }}
        .compare label {{ color: #e5e7eb; }}
        .compare input, .compare select {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); }}
        .testimonial {{
          margin-top: 22px; padding: 18px; border-radius: 16px; background: #ffffff; border: 1px solid var(--line); box-shadow: 0 14px 34px rgba(15,23,42,0.1);
          display:flex; gap: 14px; align-items:center;
        }}
        .quote-mark {{
          width: 42px; height: 42px; border-radius: 12px; background: #0f172a; color: #e5e7eb; display:flex; align-items:center; justify-content:center;
          font-size: 24px; font-weight: 800;
        }}
        .alert {{
          margin: 12px 0 18px 0; padding: 12px 14px; border-radius: 14px;
          background: #0f172a; color: #fecaca; border: 1px solid #fca5a5;
        }}
        .alert-title {{ font-weight: 800; color: #fca5a5; }}
        .alert-body {{ color: #fee2e2; line-height: 1.5; }}
        .scheme-buttons {{ display:flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }}
        .mini-btn {{
          padding: 7px 10px; border-radius: 10px; border: none; cursor: pointer;
          background: rgba(255,255,255,0.10); color: #e5e7eb; font-weight: 700;
        }}
        .scheme-btn {{ padding: 8px 10px; border-radius: 10px; border: 1px solid var(--line); background: #ffffff; color: var(--fg); font-weight: 700; cursor: pointer; }}
        .quick-grid {{ display:grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); margin-top: 8px; }}
        .quick-group {{ margin-top: 8px; }}
        .quick-label {{ font-size: 12px; font-weight: 800; color: #cbd5e1; letter-spacing: 0.03em; text-transform: uppercase; }}
        .progress-overlay {{
          display:none; position: fixed; inset: 0; z-index: 50; background: rgba(2, 6, 23, 0.78);
          backdrop-filter: blur(10px); align-items:center; justify-content:center; padding: 18px;
        }}
        .progress-card {{
          width: min(760px, 100%); background: linear-gradient(145deg, #08111f, #0f172a);
          color: #e5e7eb; border-radius: 22px; border: 1px solid rgba(255,255,255,0.10);
          box-shadow: 0 24px 60px rgba(0,0,0,0.45); padding: 20px;
        }}
        .progress-head {{ display:flex; justify-content: space-between; gap: 12px; align-items:flex-start; flex-wrap: wrap; }}
        .progress-title {{ font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }}
        .progress-subtitle {{ color: #cbd5e1; margin-top: 4px; line-height: 1.5; }}
        .progress-bar {{ margin-top: 16px; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow:hidden; }}
        .progress-fill {{ width: 0%; height: 100%; border-radius: 999px; background: linear-gradient(135deg, #5be286, #2dd4bf); transition: width 0.35s ease; }}
        .progress-fill.indeterminate {{
          width: 58%;
          transform-origin: left center;
          animation: progressPulse 1.2s ease-in-out infinite;
        }}
        @keyframes progressPulse {{
          0% {{ opacity: 0.65; transform: scaleX(0.66); }}
          50% {{ opacity: 1; transform: scaleX(1); }}
          100% {{ opacity: 0.65; transform: scaleX(0.66); }}
        }}
        .progress-steps {{ display:grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 16px; }}
        .progress-step {{
          padding: 12px 14px; border-radius: 14px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
          color: #cbd5e1; min-height: 72px;
        }}
        .progress-step.active {{ color: #ecfeff; border-color: rgba(45,212,191,0.40); box-shadow: inset 0 0 0 1px rgba(45,212,191,0.18); }}
        .progress-step b {{ display:block; color: inherit; margin-bottom: 4px; }}
        .progress-note {{ margin-top: 14px; color: #94a3b8; font-size: 13px; }}
        .modal {{
          display:none; position: fixed; inset:0; background: rgba(0,0,0,0.55); align-items:center; justify-content:center; z-index: 20; padding: 18px;
        }}
        .modal-card {{
          background:#ffffff; color: var(--fg); padding: 18px; border-radius: 16px; max-width: 720px; width: 100%;
          border: 1px solid var(--line); box-shadow: var(--shadow);
        }}
        .modal-header {{ display:flex; justify-content: space-between; align-items:center; gap:10px; }}
        .close-btn {{ border:none; background: #0f172a; color: #e5e7eb; padding: 8px 12px; border-radius: 10px; cursor: pointer; }}
        .scheme-pill {{
          padding: 6px 10px; border-radius: 10px; background: #ecfdf3; color: #166534; font-weight: 700; border: 1px solid #bbf7d0;
        }}
        @media (max-width: 960px) {{
          .hero {{ grid-template-columns: 1fr; }}
          .compare {{ grid-template-columns: 1fr; }}
          .input-row {{ grid-template-columns: 1fr; }}
          .nav-links {{ display:none; }}
          .nav-inner {{ justify-content: space-between; }}
        }}
        @media (max-width: 640px) {{
          h1 {{ font-size: 28px; }}
          .brand-badge {{ width: 30px; height: 30px; }}
        }}
      </style>
    </head>
    <body>
      <div class="nav">
        <div class="nav-inner">
          <div class="brand">
            <div class="brand-badge">RB</div>
            Riders Bay SaddleFit
          </div>
          <div class="nav-links">
            <a href="#compare">Compare</a>
          </div>
          <div class="nav-cta">
            <button class="ghost" type="button" onclick="openHorseGuide()">Horse guide</button>
            <button class="ghost" type="button" onclick="openDisciplineGuide()">Discipline guide</button>
            <a class="btn" href="#start">Launch analysis</a>
          </div>
        </div>
      </div>

      <main class="container">
        {dep_warning}
        <section class="hero" id="start">
          <div class="hero-copy">
            <span class="eyebrow">Riders Bay workspace</span>
            <h1>Riders Bay video saddle-fit lab</h1>
            <p class="lede">Upload a ride and get instant topline detection, gear checks, and rider balance metrics. Built to keep uploads and side-by-side comparisons front and center.</p>
            <div class="hero-tags">
              <span class="tag">Fast video ingest</span>
              <span class="tag">Auto-detect or calibrate</span>
              <span class="tag">PDF + comparisons</span>
            </div>
            <div class="actions">
              <a class="btn" href="#start">Upload a ride</a>
              <a class="link" href="#compare">Compare two rides -></a>
            </div>
          </div>
          <div class="start-card">
            <h3 style="margin:0;">Start analysis</h3>
            <p class="hint" style="margin-top:4px;">Designed for quick uploads and clean rider/horse metrics without unnecessary clutter.</p>
            <form id="uploadForm" action="/start" method="post" enctype="multipart/form-data">
              <label>Video (MP4 / MOV)</label>
              <input type="file" name="video" accept="video/mp4,video/quicktime,.mp4,.mov" required />

              <div class="input-row">
                <div>
                  <label>Horse profile</label>
                  <select name="horse_profile" id="horseProfileSelect">
                    {horse_profile_select_html("high_wither")}
                  </select>
                  <input type="hidden" name="horse_scheme" id="horseScheme" value="high_wither" />
                  <div class="hint">
                    <button type="button" class="mini-btn" onclick="openHorseGuide()">Profile guide</button>
                    <span style="margin-left:6px;">Quick set:</span>
                    <div class="scheme-buttons">
                      <button type="button" class="mini-btn" onclick="quickProfile('high_wither')">High wither</button>
                      <button type="button" class="mini-btn" onclick="quickProfile('round_barrel')">Round barrel</button>
                      <button type="button" class="mini-btn" onclick="quickProfile('wide_build')">Wide build</button>
                      <button type="button" class="mini-btn" onclick="quickProfile('short_back')">Short back</button>
                    </div>
                  </div>
                </div>
                <div>
                  <label>Discipline</label>
                  <select name="discipline" id="disciplineSelect">
                    {discipline_select_html("general_riding")}
                  </select>
                  <div class="hint">
                    <button type="button" class="mini-btn" onclick="openDisciplineGuide()">Discipline guide</button>
                    <span style="margin-left:6px;">Quick set:</span>
                    <div class="scheme-buttons">
                      <button type="button" class="mini-btn" onclick="quickDiscipline('general_riding')">General</button>
                      <button type="button" class="mini-btn" onclick="quickDiscipline('trail_riding')">Trail</button>
                      <button type="button" class="mini-btn" onclick="quickDiscipline('dressage')">Dressage</button>
                      <button type="button" class="mini-btn" onclick="quickDiscipline('show_jumping')">Jumping</button>
                    </div>
                  </div>
                </div>
                <div>
                  <label>Saddle type</label>
                  <select name="saddle_type">
                    <option value="english" selected>English</option>
                    <option value="western">Western</option>
                  </select>
                  <div class="hint">Fits the analytics to your saddle profile.</div>
                </div>
              </div>

              <div class="hint">
                Recording tips: stable camera, horse fully visible, side-view, 30-60 seconds, good light. Gear and safety items are auto-detected and included in the PDF.
              </div>

              <div class="hint">
                Points are detected automatically. Need the manual order? <a href="/point-selection-guide" target="_blank" style="color:#a5f3fc;">See guide.</a>
              </div>

              <button class="primary" type="submit">Upload & Auto-Analyze -></button>
            </form>
          </div>
        </section>

        <section class="section" id="compare">
          <div class="compare">
            <div>
              <h3>Compare two rides</h3>
              <p class="muted">Drop two side-view clips to see progress with the same overlay, scores, and PDF export.</p>
              <div class="hero-tags" style="margin-top:12px;">
                <span class="tag">Dual video ingest</span>
                <span class="tag">Shared profile + discipline</span>
                <span class="tag">Progress callouts</span>
              </div>
            </div>
            <form id="compareForm" action="/compare_start" method="post" enctype="multipart/form-data">
              <label>Video A (MP4 / MOV)</label>
              <input type="file" name="video_a" accept="video/mp4,video/quicktime,.mp4,.mov" required />
              <label>Video B (MP4 / MOV)</label>
              <input type="file" name="video_b" accept="video/mp4,video/quicktime,.mp4,.mov" required />

              <div class="input-row">
                <div>
                  <label>Horse profile</label>
                  <select name="horse_profile_compare" id="horseProfileCompare">
                    {horse_profile_select_html("high_wither")}
                  </select>
                  <input type="hidden" name="horse_scheme_compare" id="horseSchemeCompare" value="high_wither" />
                  <div class="hint">
                    <button type="button" class="mini-btn" onclick="openHorseGuide()">Profile guide</button>
                    <div class="scheme-buttons">
                      <button type="button" class="mini-btn" onclick="quickProfile('high_wither', true)">High wither</button>
                      <button type="button" class="mini-btn" onclick="quickProfile('round_barrel', true)">Round barrel</button>
                      <button type="button" class="mini-btn" onclick="quickProfile('wide_build', true)">Wide build</button>
                    </div>
                  </div>
                </div>
                <div>
                  <label>Discipline</label>
                  <select name="discipline_compare" id="disciplineCompare">
                    {discipline_select_html("general_riding")}
                  </select>
                  <div class="hint">
                    <button type="button" class="mini-btn" onclick="openDisciplineGuide()">Discipline guide</button>
                    <div class="scheme-buttons">
                      <button type="button" class="mini-btn" onclick="quickDiscipline('general_riding', true)">General</button>
                      <button type="button" class="mini-btn" onclick="quickDiscipline('trail_riding', true)">Trail</button>
                      <button type="button" class="mini-btn" onclick="quickDiscipline('show_jumping', true)">Jumping</button>
                    </div>
                  </div>
                </div>
                <div>
                  <label>Saddle type</label>
                  <select name="saddle_type_compare">
                    <option value="english" selected>English</option>
                    <option value="western">Western</option>
                  </select>
                </div>
              </div>

              <div class="hint">Both videos are auto-detected; we generate a side-by-side report with the same clean Riders Bay styling.</div>
              <button class="primary" type="submit" style="margin-top:12px;">Upload & Compare -></button>
            </form>
          </div>
        </section>
      </main>

      <div id="progressOverlay" class="progress-overlay" aria-hidden="true">
        <div class="progress-card" role="status" aria-live="polite">
          <div class="progress-head">
            <div>
              <div class="progress-title" id="progressTitle">Uploading video</div>
              <div class="progress-subtitle" id="progressSubtitle">Preparing the ride for rider, horse, and saddle analysis.</div>
            </div>
            <div class="scheme-pill" id="progressPercent">0%</div>
          </div>
          <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
          <div class="progress-steps" id="progressSteps"></div>
          <div class="progress-note">Keep this tab open while the analysis runs.</div>
        </div>
      </div>

      <div id="schemeModal" class="modal">
        <div class="modal-card">
          <div class="modal-header">
            <h3 style="margin:0;">Scheme guide</h3>
            <button class="close-btn" onclick="closeSchemeGuide()">Close</button>
          </div>
          <div class="muted" style="margin-top:6px;">
            Choose based on discipline and horse build:
            <ul>
              <li>Trail / Leisure: trail</li>
              <li>Arena flatwork: dressage / high_wither if prominent withers</li>
              <li>Speed / Polo / Mounted archery: polo or mounted_archery</li>
              <li>Jumping: show_jumping or eventing</li>
              <li>Body types: round_barrel (wide), narrow_build, wide_build, short_back, long_back</li>
            </ul>
          </div>
          <div class="scheme-buttons" style="margin-top:8px;">
            <button class="scheme-btn" onclick="quickScheme('trail')">Trail</button>
            <button class="scheme-btn" onclick="quickScheme('dressage')">Dressage</button>
            <button class="scheme-btn" onclick="quickScheme('racing')">Speed</button>
            <button class="scheme-btn" onclick="quickScheme('show_jumping')">Jumping</button>
          </div>
        </div>
      </div>

      <script>
        window.__SADDLEFIT_CONFIG__ = {dashboard_config};
      </script>
      <script src="/dashboard.js" defer></script>
    </body>
    </html>
    """


def _validate_model(model_cls, payload):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    return model_cls.parse_obj(payload)  # type: ignore[attr-defined]


def _analysis_scored_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scores": dict(payload.get("scores", {}) or {}),
        "flags": list(payload.get("flags", []) or []),
        "recommendations": list(payload.get("recommendations", []) or []),
        "coach": dict(payload.get("coach", {}) or {}),
        "gear_assessment": dict(payload.get("gear_assessment", {}) or {}),
        "gear_detection": dict(payload.get("gear_detection", {}) or {}),
        "gear_used": dict(payload.get("gear_used", {}) or {}),
        "quality": dict(payload.get("quality", {}) or {}),
        "pose_summary": dict(payload.get("pose_summary", {}) or {}),
        "analysis_payload": payload,
    }


def _analysis_meta_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    video_meta = payload.get("video_metadata", {}) or {}
    horse_profile = payload.get("horse_profile", payload.get("horse_scheme", "high_wither"))
    discipline = payload.get("discipline", "general_riding")
    saddle_type = payload.get("saddle_type", "english")
    return {
        "analysis_id": payload.get("analysis_id", ""),
        "video_filename": video_meta.get("filename", ""),
        "horse_scheme": horse_profile,
        "horse_profile": horse_profile,
        "discipline": discipline,
        "saddle_type": saddle_type,
        "frame_width": video_meta.get("width"),
        "frame_height": video_meta.get("height"),
    }


def _comparison_scored_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _analysis_scored_from_payload(payload)


def load_analysis_record(analysis_id: str) -> Optional[Dict[str, Any]]:
    run_dir = storage.analysis_output_dir(analysis_id)
    legacy_run_dir = LEGACY_OUTPUTS_DIR / analysis_id
    meta_path = first_existing_path(run_dir / "meta.json", legacy_run_dir / "meta.json")
    result_path = first_existing_path(run_dir / "result.json", legacy_run_dir / "result.json")
    report_path = first_existing_path(
        storage.analysis_report_dir(analysis_id) / "report.html",
        run_dir / "report.html",
        legacy_run_dir / "report.html",
    )
    if result_path is None:
        return None

    meta: Dict[str, Any] = {}
    if meta_path is not None:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    with open(result_path, "r", encoding="utf-8") as handle:
        result = json.load(handle)

    payload = result if isinstance(result, dict) and "rider_metrics" in result else None
    if payload is None:
        payload = build_analysis_payload(
            analysis_id,
            meta,
            result.get("metrics", {}) or {},
            _analysis_scored_from_payload(result),
            points=result.get("calibration_points", result.get("points", {})) or {},
            pose_summary=result.get("pose_summary", {}) or {},
            mark_scores=result.get("marks", {}) or result.get("mark_scores", {}) or {},
            created_at=result.get("created_at"),
        )
    payload = dict(payload)
    payload.setdefault("analysis_id", analysis_id)
    payload.setdefault("metrics", result.get("metrics", {}) or {})
    payload.setdefault("mark_scores", result.get("marks", {}) or result.get("mark_scores", {}) or {})
    payload.setdefault("calibration_points", result.get("calibration_points", result.get("points", {})) or {})
    payload.setdefault("points", payload.get("calibration_points", {}))
    payload.setdefault("pose_summary", result.get("pose_summary", {}) or {})
    payload.setdefault("analysis_sections_html", payload.get("analysis_sections_html", "") or build_analysis_sections_html(payload))

    record = {
        "analysis_id": analysis_id,
        "meta": meta or _analysis_meta_from_payload(payload),
        "result": result,
        "analysis_payload": payload,
        "scored": _analysis_scored_from_payload(payload),
        "mark_scores": payload.get("mark_scores", {}) or result.get("marks", {}) or {},
        "report_html": report_path.read_text(encoding="utf-8") if report_path is not None else "",
    }
    return record


def _comparison_meta_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    horse_profile = payload.get("horse_profile", "high_wither")
    discipline = payload.get("discipline", "general_riding")
    saddle_type = payload.get("saddle_type", "english")
    return {
        "horse_scheme": horse_profile,
        "horse_profile": horse_profile,
        "discipline": discipline,
        "saddle_type": saddle_type,
    }


def load_comparison_record(compare_id: str) -> Optional[Dict[str, Any]]:
    compare_path = first_existing_path(
        storage.comparison_output_dir(compare_id) / "compare.json",
        LEGACY_OUTPUTS_DIR / "compare" / compare_id / "compare.json",
    )
    compare_data: Dict[str, Any] = {}
    if compare_path is not None:
        with open(compare_path, "r", encoding="utf-8") as handle:
            compare_data = json.load(handle)

    label_a = compare_data.get("label_a") or compare_data.get("video_a", {}).get("file") or "Ride A"
    label_b = compare_data.get("label_b") or compare_data.get("video_b", {}).get("file") or "Ride B"

    analysis_a = compare_data.get("analysis_a")
    analysis_b = compare_data.get("analysis_b")
    if not isinstance(analysis_a, dict):
        record_a = load_analysis_record(f"{compare_id}_a")
        analysis_a = record_a["analysis_payload"] if record_a is not None else None
    if not isinstance(analysis_b, dict):
        record_b = load_analysis_record(f"{compare_id}_b")
        analysis_b = record_b["analysis_payload"] if record_b is not None else None
    if not isinstance(analysis_a, dict) or not isinstance(analysis_b, dict):
        return None

    comparison_payload = compare_data.get("comparison_payload")
    meta = _comparison_meta_from_payload(comparison_payload or compare_data)
    if isinstance(analysis_a, dict):
        meta["horse_profile"] = analysis_a.get("horse_profile", meta["horse_profile"])
        meta["horse_scheme"] = meta["horse_profile"]
        meta["discipline"] = analysis_a.get("discipline", meta["discipline"])
        meta["saddle_type"] = analysis_a.get("saddle_type", meta["saddle_type"])
    elif isinstance(analysis_b, dict):
        meta["horse_profile"] = analysis_b.get("horse_profile", meta["horse_profile"])
        meta["horse_scheme"] = meta["horse_profile"]
        meta["discipline"] = analysis_b.get("discipline", meta["discipline"])
        meta["saddle_type"] = analysis_b.get("saddle_type", meta["saddle_type"])
    if not isinstance(comparison_payload, dict) or not comparison_payload.get("comparisons"):
        comparison_payload = build_comparison_payload(compare_id, meta, label_a, label_b, analysis_a, analysis_b)
    comparison_payload = dict(comparison_payload)
    comparison_payload.setdefault("comparison_id", compare_id)
    comparison_payload.setdefault("created_at", compare_data.get("created_at", datetime.now(timezone.utc).isoformat()))

    record = {
        "compare_id": compare_id,
        "meta": meta,
        "label_a": label_a,
        "label_b": label_b,
        "comparison_payload": comparison_payload,
        "analysis_a": analysis_a,
        "analysis_b": analysis_b,
        "metrics_a": analysis_a.get("metrics", {}) or {},
        "metrics_b": analysis_b.get("metrics", {}) or {},
        "scored_a": _comparison_scored_from_payload(analysis_a),
        "scored_b": _comparison_scored_from_payload(analysis_b),
        "mark_scores_a": analysis_a.get("mark_scores", {}) or analysis_a.get("marks", {}) or {},
        "mark_scores_b": analysis_b.get("mark_scores", {}) or analysis_b.get("marks", {}) or {},
        "report_html": "",
        "compare_data": compare_data,
    }
    report_path = first_existing_path(
        storage.comparison_report_dir(compare_id) / "report.html",
        storage.comparison_output_dir(compare_id) / "report.html",
        LEGACY_OUTPUTS_DIR / "compare" / compare_id / "report.html",
    )
    if report_path is not None:
        record["report_html"] = report_path.read_text(encoding="utf-8")
    return record


def render_analysis_report_from_record(record: Dict[str, Any], pdf_available: bool = True) -> str:
    payload = record["analysis_payload"]
    meta = record.get("meta") or _analysis_meta_from_payload(payload)
    metrics = payload.get("metrics", {}) or record.get("result", {}).get("metrics", {}) or {}
    scored = record.get("scored") or _analysis_scored_from_payload(payload)
    points = payload.get("points", {}) or payload.get("calibration_points", {}) or {}
    mark_scores = record.get("mark_scores") or payload.get("mark_scores", {}) or record.get("result", {}).get("marks", {}) or {}
    stick_svg = build_stick_svg(metrics, points)
    growth_svg = build_growth_svg(metrics, scored.get("scores", {}))
    mark_chart_svg = build_mark_chart(mark_scores)
    return render_report_html(
        payload.get("analysis_id", record.get("analysis_id", "")),
        meta,
        metrics,
        scored,
        stick_svg,
        growth_svg,
        mark_chart_svg,
        pdf_available=pdf_available,
        analysis_payload=payload,
        points=points,
        mark_scores=mark_scores,
    )


def render_comparison_report_from_record(
    record: Dict[str, Any],
    pdf_available: bool = True,
    show_video: bool = True,
    include_analysis: bool = False,
) -> str:
    comparison_payload = record["comparison_payload"]
    compare_id = record.get("compare_id", comparison_payload.get("comparison_id", ""))
    meta = record.get("meta") or _comparison_meta_from_payload(comparison_payload)
    return render_compare_report(
        compare_id,
        meta,
        record.get("label_a", "Ride A"),
        record.get("label_b", "Ride B"),
        record.get("metrics_a", {}) or {},
        record.get("metrics_b", {}) or {},
        record.get("scored_a") or _comparison_scored_from_payload(record["analysis_a"]),
        record.get("scored_b") or _comparison_scored_from_payload(record["analysis_b"]),
        pdf_available=pdf_available,
        comparison_payload=comparison_payload,
        show_video=show_video,
        include_analysis=include_analysis,
        mark_scores_a=record.get("mark_scores_a") or {},
        mark_scores_b=record.get("mark_scores_b") or {},
    )


def process_single_analysis_path(
    video_path: Path,
    analysis_id: str,
    original_filename: str,
    horse_profile: str,
    discipline: str,
    horse_scheme: str,
    saddle_type: str,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    storage.ensure_runtime_directories()
    run_dir = storage.analysis_output_dir(analysis_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise ValueError("Video file not found.")
    if video_path.stat().st_size > get_video_upload_max_bytes():
        raise ValueError("Video exceeds the configured maximum upload size.")

    frame_path = run_dir / "frame0.png"
    w, h = extract_first_frame(video_path, frame_path)
    profile_key, discipline_key = split_legacy_scheme(horse_scheme)
    horse_profile_key = normalize_horse_profile(horse_profile or profile_key or horse_scheme)
    discipline_key = normalize_discipline(discipline or discipline_key)
    meta = {
        "analysis_id": analysis_id,
        "video_filename": video_path.name,
        "original_filename": original_filename or video_path.name,
        "horse_scheme": horse_profile_key,
        "horse_profile": horse_profile_key,
        "discipline": discipline_key,
        "saddle_type": saddle_type,
        "gear": {},
        "gear_detection": {},
        "gear_used": {},
        "frame_width": w,
        "frame_height": h,
    }
    if source_meta:
        meta.update(source_meta)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    result, metrics, scored, confidence = analyze_video_auto(run_dir, analysis_id, meta)
    points = result.get("calibration_points", {}) or {}
    pdf_generated, final_html = save_report_assets(run_dir, analysis_id, meta, metrics, scored, points)
    record = load_analysis_record(analysis_id) or {}
    record.update(
        {
            "analysis_id": analysis_id,
            "run_dir": os.fspath(run_dir),
            "meta": meta,
            "result": result,
            "analysis_payload": result,
            "scored": scored,
            "mark_scores": result.get("mark_scores", result.get("marks", {})) or {},
            "report_html": final_html,
            "pdf_generated": pdf_generated,
            "confidence": confidence,
        }
    )
    return record


def process_comparison_paths(
    video_a_path: Path,
    video_b_path: Path,
    compare_id: str,
    horse_profile: str,
    discipline: str,
    horse_scheme: str,
    saddle_type: str,
    label_a: str,
    label_b: str,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    storage.ensure_runtime_directories()
    base_dir = storage.comparison_output_dir(compare_id)
    run_a = storage.comparison_case_dir(compare_id, "a")
    run_b = storage.comparison_case_dir(compare_id, "b")
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    if not video_a_path.exists() or not video_b_path.exists():
        raise ValueError("Comparison videos are missing.")
    if video_a_path.stat().st_size > get_video_upload_max_bytes() or video_b_path.stat().st_size > get_video_upload_max_bytes():
        raise ValueError("One of the comparison videos exceeds the configured maximum upload size.")

    wa, ha = extract_first_frame(video_a_path, run_a / "frame0.png")
    wb, hb = extract_first_frame(video_b_path, run_b / "frame0.png")

    profile_key, discipline_key = split_legacy_scheme(horse_scheme)
    horse_profile_key = normalize_horse_profile(horse_profile or profile_key or horse_scheme)
    discipline_key = normalize_discipline(discipline or discipline_key)
    meta_base = {
        "horse_scheme": horse_profile_key,
        "horse_profile": horse_profile_key,
        "discipline": discipline_key,
        "saddle_type": saddle_type,
    }
    meta_a = {
        **meta_base,
        "analysis_id": compare_id + "_a",
        "video_filename": video_a_path.name,
        "original_filename": label_a or video_a_path.name,
        "frame_width": wa,
        "frame_height": ha,
    }
    meta_b = {
        **meta_base,
        "analysis_id": compare_id + "_b",
        "video_filename": video_b_path.name,
        "original_filename": label_b or video_b_path.name,
        "frame_width": wb,
        "frame_height": hb,
    }
    if source_meta:
        meta_a.update(source_meta.get("video_a", {}) if isinstance(source_meta.get("video_a"), dict) else {})
        meta_b.update(source_meta.get("video_b", {}) if isinstance(source_meta.get("video_b"), dict) else {})

    result_a, metrics_a, scored_a, conf_a = analyze_video_auto(run_a, meta_a["analysis_id"], meta_a)
    result_b, metrics_b, scored_b, conf_b = analyze_video_auto(run_b, meta_b["analysis_id"], meta_b)

    compare_meta = {"horse_scheme": horse_profile_key, "horse_profile": horse_profile_key, "discipline": discipline_key, "saddle_type": saddle_type}
    comparison_payload = build_comparison_payload(compare_id, compare_meta, label_a, label_b, result_a, result_b)
    pdf_generated, final_html = save_compare_report(
        base_dir,
        compare_id,
        compare_meta,
        label_a,
        label_b,
        metrics_a,
        metrics_b,
        scored_a,
        scored_b,
        comparison_payload=comparison_payload,
    )

    compare_json = {
        "compare_id": compare_id,
        "created_at": comparison_payload.get("created_at"),
        "label_a": label_a,
        "label_b": label_b,
        "comparison_payload": comparison_payload,
        "analysis_a": result_a,
        "analysis_b": result_b,
        "video_a": {"file": label_a, "confidence": conf_a, "metrics": metrics_a, "scores": scored_a["scores"]},
        "video_b": {"file": label_b, "confidence": conf_b, "metrics": metrics_b, "scores": scored_b["scores"]},
    }
    with open(base_dir / "compare.json", "w", encoding="utf-8") as handle:
        json.dump(compare_json, handle, indent=2)

    record = load_comparison_record(compare_id) or {}
    record.update(
        {
            "compare_id": compare_id,
            "base_dir": os.fspath(base_dir),
            "meta": compare_meta,
            "label_a": label_a,
            "label_b": label_b,
            "comparison_payload": comparison_payload,
            "analysis_a": result_a,
            "analysis_b": result_b,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "scored_a": scored_a,
            "scored_b": scored_b,
            "mark_scores_a": compute_mark_scores(metrics_a, scored_a),
            "mark_scores_b": compute_mark_scores(metrics_b, scored_b),
            "report_html": final_html,
            "pdf_generated": pdf_generated,
            "compare_json": compare_json,
        }
    )
    return record


async def process_single_analysis_upload(
    video: UploadFile,
    horse_profile: str,
    discipline: str,
    horse_scheme: str,
    saddle_type: str,
) -> Dict[str, Any]:
    analysis_id = str(uuid.uuid4())
    run_dir = storage.analysis_output_dir(analysis_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_video_filename(video.filename or "uploaded_video.mp4")
    video_path = run_dir / filename
    await write_uploadfile_to_path(video, video_path)

    return process_single_analysis_path(
        video_path,
        analysis_id,
        video.filename or filename,
        horse_profile,
        discipline,
        horse_scheme,
        saddle_type,
        source_meta={"upload_mode": "multipart"},
    )


async def process_single_analysis_reference(
    payload: AnalysisRequest,
) -> Dict[str, Any]:
    analysis_id = str(uuid.uuid4())
    run_dir = storage.analysis_output_dir(analysis_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_url = validate_video_reference_url(payload.video_url or "")
    original_filename = sanitize_video_filename(payload.original_filename or os.path.basename(urlparse(source_url).path) or "uploaded_video.mp4")
    video_path = run_dir / original_filename
    download_blob_to_path(source_url, video_path)

    return process_single_analysis_path(
        video_path,
        analysis_id,
        payload.original_filename or original_filename,
        payload.horse_profile,
        payload.discipline,
        "",
        payload.saddle_type,
        source_meta={
            "video_url": source_url,
            "storage_key": payload.storage_key or "",
            "storage_provider": payload.storage_provider or "vercel_blob",
            "upload_mode": "direct",
        },
    )


async def process_comparison_uploads(
    video_a: UploadFile,
    video_b: UploadFile,
    horse_profile: str,
    discipline: str,
    horse_scheme: str,
    saddle_type: str,
) -> Dict[str, Any]:
    compare_id = str(uuid.uuid4())
    run_a = storage.comparison_case_dir(compare_id, "a")
    run_b = storage.comparison_case_dir(compare_id, "b")
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    fname_a = sanitize_video_filename(video_a.filename or "video_a.mp4")
    fname_b = sanitize_video_filename(video_b.filename or "video_b.mp4")
    path_a = run_a / fname_a
    path_b = run_b / fname_b
    await write_uploadfile_to_path(video_a, path_a)
    await write_uploadfile_to_path(video_b, path_b)

    return process_comparison_paths(
        path_a,
        path_b,
        compare_id,
        horse_profile,
        discipline,
        horse_scheme,
        saddle_type,
        video_a.filename or fname_a,
        video_b.filename or fname_b,
        source_meta={"upload_mode": "multipart"},
    )


async def process_comparison_reference(
    payload: ComparisonRequest,
) -> Dict[str, Any]:
    compare_id = str(uuid.uuid4())
    run_a = storage.comparison_case_dir(compare_id, "a")
    run_b = storage.comparison_case_dir(compare_id, "b")
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    source_url_a = validate_video_reference_url(payload.video_a_url or "")
    source_url_b = validate_video_reference_url(payload.video_b_url or "")
    fname_a = sanitize_video_filename(payload.video_a_filename or os.path.basename(urlparse(source_url_a).path) or "video_a.mp4")
    fname_b = sanitize_video_filename(payload.video_b_filename or os.path.basename(urlparse(source_url_b).path) or "video_b.mp4")
    path_a = run_a / fname_a
    path_b = run_b / fname_b
    download_blob_to_path(source_url_a, path_a)
    download_blob_to_path(source_url_b, path_b)

    return process_comparison_paths(
        path_a,
        path_b,
        compare_id,
        payload.horse_profile,
        payload.discipline,
        "",
        payload.saddle_type,
        payload.video_a_filename or fname_a,
        payload.video_b_filename or fname_b,
        source_meta={
            "video_a": {
                "video_url": source_url_a,
                "storage_key": payload.video_a_key or "",
                "storage_provider": payload.storage_provider or "vercel_blob",
            },
            "video_b": {
                "video_url": source_url_b,
                "storage_key": payload.video_b_key or "",
                "storage_provider": payload.storage_provider or "vercel_blob",
            },
            "upload_mode": "direct",
        },
    )


if MULTIPART_AVAILABLE:
    @app.post("/start", response_class=HTMLResponse)
    async def start(
        video: UploadFile = File(...),
        horse_profile: str = Form("high_wither"),
        discipline: str = Form("general_riding"),
        horse_scheme: str = Form(""),
        saddle_type: str = Form("english"),
    ):
        if video is None or not video.filename:
            return HTMLResponse("<h3>No video uploaded.</h3>", status_code=400)
        try:
            record = await process_single_analysis_upload(video, horse_profile, discipline, horse_scheme, saddle_type)
        except Exception as exc:
            return HTMLResponse(f"""
            <html><body style="font-family: Arial; margin: 32px;">
              <h3>Auto-detection failed: {exc}</h3>
              <p>Please try a clearer side-view video, or re-upload and calibrate manually after the new analysis id is created.</p>
              <p><a href="/">Back to home</a></p>
            </body></html>
            """, status_code=400)
        return HTMLResponse(record.get("report_html") or render_analysis_report_from_record(record, pdf_available=bool(record.get("pdf_generated"))))


    @app.post("/compare_start", response_class=HTMLResponse)
    async def compare_start(
        video_a: UploadFile = File(...),
        video_b: UploadFile = File(...),
        horse_profile_compare: str = Form("high_wither"),
        discipline_compare: str = Form("general_riding"),
        horse_scheme_compare: str = Form(""),
        saddle_type_compare: str = Form("english"),
    ):
        if not video_a.filename or not video_b.filename:
            return HTMLResponse("<h3>Both videos are required.</h3>", status_code=400)
        try:
            record = await process_comparison_uploads(
                video_a,
                video_b,
                horse_profile_compare,
                discipline_compare,
                horse_scheme_compare,
                saddle_type_compare,
            )
        except Exception as exc:
            return HTMLResponse(f"<h3>Comparison failed:</h3><p>{exc}</p>", status_code=400)
        return HTMLResponse(record.get("report_html") or render_comparison_report_from_record(record, pdf_available=bool(record.get("pdf_generated")), show_video=True, include_analysis=False))
else:
    @app.post("/start", response_class=HTMLResponse)
    async def start_missing():
        return HTMLResponse("<h3>Server missing dependency python-multipart. Please install it with 'pip install python-multipart' and restart.</h3>", status_code=500)

    @app.post("/compare_start", response_class=HTMLResponse)
    async def compare_start_missing():
        return HTMLResponse("<h3>Server missing dependency python-multipart. Please install it with 'pip install python-multipart' and restart.</h3>", status_code=500)


@app.post("/api/analyze", response_model=AnalysisResponse)
async def api_analyze(payload: AnalysisRequest = Body(...)):
    if not payload.video_url:
        return JSONResponse({"error": "Missing video_url."}, status_code=400)
    try:
        record = await process_single_analysis_reference(payload)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        return JSONResponse({"error": "Analysis failed."}, status_code=500)
    response_payload = dict(record["analysis_payload"])
    response_payload["report_html"] = None
    return _validate_model(AnalysisResponse, response_payload)


@app.get("/api/analysis/{analysis_id}", response_model=AnalysisResponse)
def api_analysis_get(analysis_id: str):
    record = load_analysis_record(analysis_id)
    if record is None:
        return JSONResponse({"error": "analysis_not_found"}, status_code=404)
    payload = dict(record["analysis_payload"])
    payload["report_html"] = None
    return _validate_model(AnalysisResponse, payload)


@app.get("/api/analysis/{analysis_id}/report", response_model=AnalysisReportResponse)
def api_analysis_report(analysis_id: str):
    record = load_analysis_record(analysis_id)
    if record is None:
        return JSONResponse({"error": "analysis_not_found"}, status_code=404)
    report_html = record.get("report_html") or render_analysis_report_from_record(record, pdf_available=True)
    payload = dict(record["analysis_payload"])
    payload["report_html"] = None
    analysis_model = _validate_model(AnalysisResponse, payload)
    return _validate_model(
        AnalysisReportResponse,
        {
            "analysis": analysis_model,
            "report_html": report_html,
            "report_url": payload.get("report_url"),
            "pdf_url": payload.get("pdf_url"),
        },
    )


@app.post("/api/compare", response_model=ComparisonResponse)
async def api_compare(payload: ComparisonRequest = Body(...)):
    if not payload.video_a_url or not payload.video_b_url:
        return JSONResponse({"error": "Both video_a_url and video_b_url are required."}, status_code=400)
    try:
        record = await process_comparison_reference(payload)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        return JSONResponse({"error": "Comparison failed."}, status_code=500)
    response_payload = dict(record["comparison_payload"])
    response_payload["report_html"] = None
    return _validate_model(ComparisonResponse, response_payload)


@app.get("/api/comparison/{comparison_id}", response_model=ComparisonResponse)
def api_comparison_get(comparison_id: str):
    record = load_comparison_record(comparison_id)
    if record is None:
        return JSONResponse({"error": "comparison_not_found"}, status_code=404)
    payload = dict(record["comparison_payload"])
    payload["report_html"] = None
    return _validate_model(ComparisonResponse, payload)


@app.get("/api/comparison/{comparison_id}/report", response_model=ComparisonReportResponse)
def api_comparison_report(comparison_id: str):
    record = load_comparison_record(comparison_id)
    if record is None:
        return JSONResponse({"error": "comparison_not_found"}, status_code=404)
    report_html = record.get("report_html") or render_comparison_report_from_record(record, pdf_available=True, show_video=True, include_analysis=False)
    payload = dict(record["comparison_payload"])
    payload["report_html"] = None
    comparison_model = _validate_model(ComparisonResponse, payload)
    return _validate_model(
        ComparisonReportResponse,
        {
            "comparison": comparison_model,
            "report_html": report_html,
            "report_url": payload.get("report_url"),
            "pdf_url": payload.get("pdf_url"),
        },
    )


@app.get("/frame/{analysis_id}")
def get_frame(analysis_id: str):
    path = first_existing_path(
        storage.analysis_output_dir(analysis_id) / "frame0.png",
        LEGACY_OUTPUTS_DIR / analysis_id / "frame0.png",
    )
    if path is None:
        return JSONResponse({"error": "frame not found"}, status_code=404)
    return FileResponse(os.fspath(path))


@app.post("/auto_calibrate/{analysis_id}")
def auto_calibrate_api(analysis_id: str):
    meta_path = first_existing_path(
        storage.analysis_output_dir(analysis_id) / "meta.json",
        LEGACY_OUTPUTS_DIR / analysis_id / "meta.json",
    )
    if meta_path is None:
        return JSONResponse({"error": "invalid analysis_id"}, status_code=404)

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    frame_path = first_existing_path(
        storage.analysis_output_dir(analysis_id) / "frame0.png",
        LEGACY_OUTPUTS_DIR / analysis_id / "frame0.png",
    )
    if frame_path is None:
        return JSONResponse({"error": "frame missing"}, status_code=404)
    frame = cv2.imread(os.fspath(frame_path))
    if frame is None:
        return JSONResponse({"error": "frame missing"}, status_code=404)

    horse_profile, discipline, saddle_type = resolve_analysis_context(meta)
    res = auto_calibrate_points(frame, horse_profile, saddle_type, discipline)
    if res["points"] is None:
        return JSONResponse({"error": "auto_calibration_failed", "details": res.get("details")}, status_code=400)
    return JSONResponse(res)


@app.get("/calibrate/{analysis_id}", response_class=HTMLResponse)
def calibrate_page(analysis_id: str):
    meta_path = first_existing_path(
        storage.analysis_output_dir(analysis_id) / "meta.json",
        LEGACY_OUTPUTS_DIR / analysis_id / "meta.json",
    )
    if meta_path is None:
        return HTMLResponse("<h3>Invalid analysis_id</h3>", status_code=404)

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    horse_profile, discipline, saddle_type = resolve_analysis_context(meta)

    # NOTE: no JS template strings with ${...} to avoid Python f-string conflicts.
    return f"""
    <html>
    <head>
      <title>Calibration - {analysis_id}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        body {{ font-family: Arial; margin: 0; background: #0b1220; color: #e5e7eb; }}
        .wrap {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
        .top {{ display:flex; justify-content: space-between; gap: 12px; align-items:center; }}
        .card {{ margin-top: 14px; padding: 16px; border-radius: 14px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); }}
        .row {{ display:flex; gap: 18px; flex-wrap: wrap; }}
        .left {{ flex: 1 1 720px; }}
        .right {{ flex: 1 1 300px; min-width: 280px; }}
        button {{
          width: 100%; margin-top: 10px; padding: 12px 14px; border-radius: 12px; border:none;
          background: linear-gradient(135deg, #f59e0b, #ef4444); color:#111827; font-weight: 800; cursor: pointer;
        }}
        .small {{ color:#cbd5e1; font-size: 13px; line-height: 1.4; }}
        canvas {{ width: 100%; border-radius: 12px; border: 1px solid rgba(255,255,255,0.16); }}
        .badge {{ display:inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(59,130,246,0.18); border: 1px solid rgba(59,130,246,0.35); }}
        .step {{ margin: 8px 0; padding: 10px; border-radius: 10px; background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.10); }}
        .ok {{ color:#22c55e; font-weight:700; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="top">
          <div>
            <h2 style="margin:0;">Calibration</h2>
            <div class="small">Click 8 points in order: <b>saddle_front</b> -> <b>saddle_rear</b> -> <b>withers</b> -> <b>croup</b> -> <b>left_shoulder</b> -> <b>right_shoulder</b> -> <b>left_hip</b> -> <b>right_hip</b>. Auto-detect is available anytime if you prefer.</div>
          </div>
          <div class="badge">ID: {analysis_id}</div>
        </div>

        <div class="row">
          <div class="left card">
            <canvas id="cv"></canvas>
            <div class="small" style="margin-top:10px;">Tip: zoom browser (Ctrl + +) if needed.</div>
          </div>

          <div class="right card">
            <div class="step"><b>Horse profile:</b> {horse_profile}</div>
            <div class="step"><b>Discipline:</b> {discipline}</div>
            <div class="step"><b>Saddle type:</b> {saddle_type}</div>

            <div class="step">
              <b>Points:</b>
              <div id="pstatus" class="small"></div>
            </div>

            <button onclick="submitPoints()">Run Tracking & Generate Report</button>
            <button style="background: linear-gradient(135deg,#22c55e,#06b6d4);" onclick="resetPoints()">Reset Points</button>
            <button style="background: linear-gradient(135deg,#60a5fa,#a78bfa);" onclick="autoDetectPoints()">Auto-detect Points</button>

            <div class="small" style="margin-top:12px;">
              Need help? <a href="/point-selection-guide" target="_blank">Open the point selection guide</a>.
            </div>
          </div>
        </div>
      </div>

      <script>
        const analysisId = "{analysis_id}";
        const order = ["saddle_front", "saddle_rear", "withers", "croup", "left_shoulder", "right_shoulder", "left_hip", "right_hip"];
        let clicks = [];

        let img = new Image();
        img.src = "/frame/" + analysisId;

        const canvas = document.getElementById("cv");
        const ctx = canvas.getContext("2d");
        const pstatus = document.getElementById("pstatus");

        function draw() {{
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          ctx.drawImage(img, 0, 0);

          clicks.forEach((pt, idx) => {{
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 10, 0, Math.PI*2);
            ctx.fillStyle = "rgba(34,197,94,0.85)";
            ctx.fill();
            ctx.lineWidth = 3;
            ctx.strokeStyle = "rgba(0,0,0,0.6)";
            ctx.stroke();
            ctx.fillStyle = "white";
            ctx.font = "20px Arial";
            ctx.fillText(String(idx+1), pt.x + 12, pt.y - 12);
          }});

          let html = "";
          for (let idx=0; idx<order.length; idx++) {{
            if (idx < clicks.length) {{
              html += "<div class='ok'>&#10003; " + (idx+1) + ". " + order[idx] +
                      " = (" + Math.round(clicks[idx].x) + ", " + Math.round(clicks[idx].y) + ")</div>";
            }} else {{
              html += "<div>&bull; " + (idx+1) + ". " + order[idx] + " (click next)</div>";
            }}
          }}
          pstatus.innerHTML = html;
        }}

        img.onload = () => draw();

        function getMousePos(evt) {{
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvas.width / rect.width;
          const scaleY = canvas.height / rect.height;
          return {{
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top) * scaleY
          }};
        }}

        canvas.addEventListener("click", (evt) => {{
          if (clicks.length >= order.length) return;
          const pos = getMousePos(evt);
          clicks.push(pos);
          draw();
        }});

        function resetPoints() {{
          clicks = [];
          draw();
        }}

        async function submitPoints() {{
          if (clicks.length < order.length) {{
            alert("Please click all points first.");
            return;
          }}

          const payload = {{
            saddle_front: [clicks[0].x, clicks[0].y],
            saddle_rear:  [clicks[1].x, clicks[1].y],
            withers:      [clicks[2].x, clicks[2].y],
            croup:        [clicks[3].x, clicks[3].y],
            left_shoulder:[clicks[4].x, clicks[4].y],
            right_shoulder:[clicks[5].x, clicks[5].y],
            left_hip:     [clicks[6].x, clicks[6].y],
            right_hip:    [clicks[7].x, clicks[7].y]
          }};

          const res = await fetch("/run/" + analysisId, {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(payload)
          }});

          const txt = await res.text();
          document.open(); document.write(txt); document.close();
        }}

        async function autoDetectPoints() {{
          const res = await fetch("/auto_calibrate/" + analysisId, {{ method: "POST" }});
          if (!res.ok) {{
            alert("Auto-detect failed. Please click points manually.");
            return;
          }}
          const data = await res.json();
          if (!data.points) {{
            alert("Auto-detect could not find points. Please click manually.");
            return;
          }}
          const pts = data.points;
          clicks = order.map((name) => {{
            const p = pts[name];
            return {{ x: p[0], y: p[1] }};
          }});
          draw();
          alert("Points auto-filled. Review and submit.");
        }}
      </script>
    </body>
    </html>
    """


@app.post("/run/{analysis_id}", response_class=HTMLResponse)
async def run_tracking(analysis_id: str, points: dict = Body(...)):
    run_dir = storage.analysis_output_dir(analysis_id)
    legacy_run_dir = LEGACY_OUTPUTS_DIR / analysis_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = first_existing_path(run_dir / "meta.json", legacy_run_dir / "meta.json")
    if meta_path is None:
        return HTMLResponse("<h3>Invalid analysis_id</h3>", status_code=404)

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    video_path = first_existing_path(run_dir / meta["video_filename"], legacy_run_dir / meta["video_filename"])
    if video_path is None:
        return HTMLResponse("<h3>Video not found</h3>", status_code=404)

    required = [
        "saddle_front",
        "saddle_rear",
        "withers",
        "croup",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]
    for k in required:
        if k not in points or not isinstance(points[k], list) or len(points[k]) != 2:
            return HTMLResponse(f"<h3>Missing/invalid point: {k}</h3>", status_code=400)

    # Normalize points (fix typing + ensure floats)
    init_points = normalize_points(points)

    # Auto-detect gear if not stored yet (manual calibration path).
    frame0_path = first_existing_path(run_dir / "frame0.png", legacy_run_dir / "frame0.png")
    frame0 = cv2.imread(os.fspath(frame0_path)) if frame0_path is not None else None
    detection_data = auto_detect_gear_and_safety(frame0, init_points) if frame0 is not None else {"gear": {}, "confidences": {}, "notes": ["Frame missing for gear detection."], "method": "frame_missing"}
    meta["gear_detection"] = detection_data
    merged_gear = merge_gear_sources(detection_data.get("gear", {}), meta.get("gear", {}))
    meta["gear"] = merged_gear
    meta["gear_used"] = merged_gear
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Extract frames and track
    frames, times = extract_frames(os.fspath(video_path), target_fps=12)
    tracks, tstats = track_points_lk(frames, init_points)
    metrics = compute_metrics(tracks, times, tstats)
    gear_assessment = evaluate_gear(merged_gear, metrics, meta["horse_scheme"])
    pose_summary = sample_rider_pose_metrics(frames, times, init_points)
    scored = score_and_recommend(
        metrics,
        meta.get("horse_profile", meta.get("horse_scheme", "high_wither")),
        meta["saddle_type"],
        meta.get("discipline", "general_riding"),
        pose_summary,
    )
    scored["gear_assessment"] = gear_assessment
    scored["gear_detection"] = detection_data
    scored["gear_used"] = merged_gear
    scored["pose_summary"] = pose_summary
    mark_scores = compute_mark_scores(metrics, scored)

    result = build_analysis_payload(
        analysis_id,
        meta,
        metrics,
        scored,
        points=init_points,
        pose_summary=pose_summary,
        mark_scores=mark_scores,
    )
    result["gear"] = meta.get("gear", {})
    result["gear_detected"] = detection_data
    result["gear_detection"] = detection_data
    result["gear_used"] = merged_gear
    result["gear_assessment"] = gear_assessment
    result["marks"] = mark_scores
    result["calibration_points"] = init_points
    result["horse_scheme"] = meta.get("horse_scheme", meta.get("horse_profile", "high_wither"))
    result["video_filename"] = meta["video_filename"]
    scored["analysis_payload"] = result

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    pdf_generated, final_html = save_report_assets(run_dir, analysis_id, meta, metrics, scored, result["calibration_points"])
    return HTMLResponse(final_html)


@app.get("/report/{analysis_id}", response_class=HTMLResponse)
def report(analysis_id: str):
    path = first_existing_path(
        storage.analysis_report_dir(analysis_id) / "report.html",
        storage.analysis_output_dir(analysis_id) / "report.html",
        LEGACY_OUTPUTS_DIR / analysis_id / "report.html",
    )
    if path is None:
        record = load_analysis_record(analysis_id)
        if record is None:
            return HTMLResponse("<h3>Report not found</h3>", status_code=404)
        return record.get("report_html") or render_analysis_report_from_record(record, pdf_available=bool(record.get("pdf_generated", True)))
    return path.read_text(encoding="utf-8")


@app.get("/report/{analysis_id}.pdf")
def report_pdf(analysis_id: str):
    run_dir = storage.analysis_output_dir(analysis_id)
    legacy_run_dir = LEGACY_OUTPUTS_DIR / analysis_id
    report_dir = storage.analysis_report_dir(analysis_id)
    pdf_path = report_dir / "report.pdf"

    # If a PDF file already exists, serve it immediately.
    existing_pdf = first_existing_path(pdf_path, run_dir / "report.pdf", legacy_run_dir / "report.pdf")
    if existing_pdf is not None:
        return FileResponse(os.fspath(existing_pdf), media_type="application/pdf", filename=f"report_{analysis_id}.pdf")

    meta_path = first_existing_path(run_dir / "meta.json", legacy_run_dir / "meta.json")
    result_path = first_existing_path(run_dir / "result.json", legacy_run_dir / "result.json")
    if meta_path is not None and result_path is not None:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        with open(result_path, "r", encoding="utf-8") as handle:
            res = json.load(handle)
        mark_scores = res.get("marks", {})
        scores_block = {
            "scores": res.get("scores", {}),
            "flags": res.get("flags", []),
            "recommendations": res.get("recommendations", []),
            "gear_assessment": res.get("gear_assessment", {}),
            "coach": res.get("coach", {}),
            "gear_detection": res.get("gear_detected", {}),
            "gear_used": res.get("gear_used", {}),
            "analysis_payload": res,
            "pose_summary": res.get("pose_summary", {}),
        }
        ensure_pdf_file(analysis_id, meta, res.get("metrics", {}), scores_block, mark_scores, pdf_path)
        return FileResponse(os.fspath(pdf_path), media_type="application/pdf", filename=f"report_{analysis_id}.pdf")

    # Fallback: if HTML exists, convert/serve it as PDF so the user sees the report.
    html_path = first_existing_path(report_dir / "report.html", run_dir / "report.html", legacy_run_dir / "report.html")
    if html_path is not None:
        storage.ensure_parent_dir(pdf_path)
        try:
            pdf_renderer = get_weasyprint()
            if pdf_renderer is not None:
                pdf_renderer.HTML(filename=os.fspath(html_path), base_url=BASE_DIR).write_pdf(os.fspath(pdf_path))
                return FileResponse(os.fspath(pdf_path), media_type="application/pdf", filename=f"report_{analysis_id}.pdf")
        except Exception:
            pass
        # Fallback: build a simple PDF pointing user to HTML content.
        pdf_bytes = build_simple_pdf_bytes(
            "Saddle Fit Report",
            [
                f"ID: {analysis_id}",
                "Full report HTML is saved on the server.",
                "PDF rendering needs 'weasyprint'. Install it to enable full PDF export.",
            ],
        )
        with open(pdf_path, "wb") as pf:
            pf.write(pdf_bytes)
        return FileResponse(os.fspath(pdf_path), media_type="application/pdf", filename=f"report_{analysis_id}.pdf")

    return HTMLResponse("<h3>Report PDF not found and no saved data to regenerate.</h3>", status_code=404)


@app.get("/report_pdf/{analysis_id}")
def report_pdf_direct(analysis_id: str):
    """
    Explicit PDF endpoint to avoid clashes with the HTML route.
    """
    return report_pdf(analysis_id)


@app.get("/compare_report/{compare_id}", response_class=HTMLResponse)
def compare_report(compare_id: str):
    record = load_comparison_record(compare_id)
    if record is None:
        return HTMLResponse("<h3>Comparison report not found</h3>", status_code=404)
    if record.get("report_html"):
        return record["report_html"]
    return render_comparison_report_from_record(record, pdf_available=bool(record.get("pdf_generated", True)), show_video=True, include_analysis=False)


@app.get("/compare_report/{compare_id}.pdf")
def compare_report_pdf(compare_id: str):
    pdf_path = first_existing_path(
        storage.comparison_report_dir(compare_id) / "report.pdf",
        storage.comparison_output_dir(compare_id) / "report.pdf",
        LEGACY_OUTPUTS_DIR / "compare" / compare_id / "report.pdf",
    )
    if pdf_path is not None:
        return FileResponse(os.fspath(pdf_path), media_type="application/pdf", filename=f"compare_{compare_id}.pdf")

    record = load_comparison_record(compare_id)
    if record is None:
        return HTMLResponse("<h3>Comparison PDF not found (install weasyprint to enable PDF export).</h3>", status_code=404)

    fallback_dir = storage.comparison_report_dir(compare_id)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fallback_dir / "report.pdf"
    try:
        pdf_renderer = get_weasyprint()
        if pdf_renderer is not None:
            html_text = record.get("report_html") or render_comparison_report_from_record(record, pdf_available=True, show_video=True, include_analysis=False)
            pdf_renderer.HTML(string=html_text, base_url=BASE_DIR).write_pdf(os.fspath(pdf_path))
            return FileResponse(os.fspath(pdf_path), media_type="application/pdf", filename=f"compare_{compare_id}.pdf")
    except Exception:
        pass

    pdf_bytes = build_simple_pdf_bytes(
        "Saddle Fit Comparison",
        [
            f"Comparison ID: {compare_id}",
            "Full comparison HTML is saved on the server.",
            "PDF rendering needs 'weasyprint'. Install it to enable full PDF export.",
        ],
    )
    with open(pdf_path, "wb") as pf:
        pf.write(pdf_bytes)
    return FileResponse(os.fspath(pdf_path), media_type="application/pdf", filename=f"compare_{compare_id}.pdf")


def guess_mime(video_path: str) -> str:
    ext = os.path.splitext(video_path)[1].lower()
    if ext in [".mp4", ".m4v"]:
        return "video/mp4"
    if ext in [".mov", ".qt"]:
        return "video/quicktime"
    return "application/octet-stream"


@app.get("/video/{analysis_id}")
def video_file(analysis_id: str):
    meta_path = first_existing_path(
        storage.analysis_output_dir(analysis_id) / "meta.json",
        LEGACY_OUTPUTS_DIR / analysis_id / "meta.json",
    )
    if meta_path is None:
        return HTMLResponse("<h3>Invalid analysis_id</h3>", status_code=404)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    video_path = first_existing_path(
        storage.analysis_output_dir(analysis_id) / meta["video_filename"],
        LEGACY_OUTPUTS_DIR / analysis_id / meta["video_filename"],
    )
    if video_path is None:
        return HTMLResponse("<h3>Video not found</h3>", status_code=404)
    return FileResponse(os.fspath(video_path), media_type=guess_mime(os.fspath(video_path)), filename=meta["video_filename"])


@app.get("/compare_video/{compare_id}/{label}")
def compare_video(compare_id: str, label: str):
    if label not in ["a", "b"]:
        return HTMLResponse("<h3>Invalid label</h3>", status_code=400)
    base_dir = storage.comparison_case_dir(compare_id, label)
    legacy_base_dir = LEGACY_OUTPUTS_DIR / "compare" / compare_id / label
    meta_path = first_existing_path(base_dir / "meta.json", legacy_base_dir / "meta.json")
    if meta_path is None:
        return HTMLResponse("<h3>Invalid comparison id</h3>", status_code=404)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    video_path = first_existing_path(base_dir / meta["video_filename"], legacy_base_dir / meta["video_filename"])
    if video_path is None:
        return HTMLResponse("<h3>Video not found</h3>", status_code=404)
    return FileResponse(os.fspath(video_path), media_type=guess_mime(os.fspath(video_path)), filename=meta["video_filename"])
