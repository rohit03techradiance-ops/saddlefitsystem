import os
import uuid
import json
import math
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import runtime_paths as storage

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


def auto_calibrate_points(frame_bgr: np.ndarray, horse_scheme: str, saddle_type: str) -> Dict:
    """
    Returns:
      {
        "points": {saddle_front:[x,y], saddle_rear:[x,y], withers:[x,y], croup:[x,y]},
        "confidence": float,
        "details": {...}
      }
    """
    h, w = frame_bgr.shape[:2]

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

    # Offsets: tune by discipline / saddle type
    # (These are MVP defaults; we will tune later using sample videos.)
    base_len = 0.10 * w  # 10% of frame width
    if saddle_type == "western":
        base_len = 0.12 * w
    if horse_scheme in ["polo", "mounted_archery", "tent_pegging", "racing", "eventing"]:
        base_len *= 1.05

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

# --------- Horse "schemes" / disciplines list ----------
HORSE_SCHEMES = [
    # Body-type schemes
    ("high_wither", "High-wither (narrow / prominent withers)"),
    ("round_barrel", "Round-barrel (girthy / wide ribcage)"),
    ("narrow_build", "Narrow build (slimmer frame)"),
    ("wide_build", "Wide build (broad back)"),
    ("short_back", "Short back (compact)"),
    ("long_back", "Long back (longer saddle support area)"),

    # Riding disciplines / horse games
    ("polo", "Polo"),
    ("mounted_archery", "Mounted Archery"),
    ("tent_pegging", "Tent Pegging"),
    ("dressage", "Dressage"),
    ("show_jumping", "Show Jumping"),
    ("eventing", "Eventing"),
    ("endurance", "Endurance / Long trail"),
    ("racing", "Racing / Speed work"),
    ("trail", "Trail / Leisure riding"),
    ("reining", "Reining (Western)"),
]


@app.get("/horse-scheme-guide", response_class=HTMLResponse)
def horse_scheme_guide():
    guide_path = os.path.join(BASE_DIR, "horse_scheme_guide.html")
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


def score_and_recommend(metrics: Dict[str, Any], horse_scheme: str, saddle_type: str) -> Dict:
    pitch = metrics["pitch_mean_deg"]
    rock = metrics["rock_amplitude_deg"]
    drift_x = metrics["mid_drift_x_px"]
    drift_rate = metrics.get("mid_drift_rate_px_s", 0.0)
    bounce = metrics.get("mid_bounce_y_px", 0.0)
    shoulder_std = metrics["shoulder_level_std_px"]
    hip_std = metrics["hip_level_std_px"]
    cadence = metrics["cadence_hz"]
    clearance_collapse = metrics.get("alignment", {}).get("clearance_collapse", False)
    clearance_min = metrics.get("alignment", {}).get("withers_clearance_min_px", 0.0)
    bridging_proxy = metrics.get("alignment", {}).get("bridging_proxy", 0.0)
    tracking_conf = metrics.get("tracking", {}).get("tracking_success_pct", 0.0)

    front_down_threshold = -3.0 if horse_scheme in ["high_wither", "dressage", "show_jumping"] else -5.0
    rock_threshold = 6.0 if saddle_type == "english" else 8.0

    if horse_scheme in ["polo", "mounted_archery", "tent_pegging", "racing", "eventing"]:
        rock_threshold += 1.5

    drift_threshold = 45.0 if horse_scheme in ["round_barrel", "polo", "mounted_archery", "tent_pegging"] else 35.0
    align_threshold = 12.0
    bounce_threshold = 24.0

    flags: List[str] = []
    recs: List[str] = []

    if pitch < front_down_threshold:
        flags.append("Front-down pitch trend detected (saddle may be tipping forward).")
        recs.append("Check withers clearance + front balance. Consider fitter check or front pad/shim if appropriate.")

    if rock > rock_threshold:
        flags.append("Higher saddle rocking detected (oscillation over stride).")
        recs.append("Check panel contact (bridging risk). Try different girthing, pad, or consult saddle fitter.")

    if drift_x > drift_threshold:
        flags.append("Noticeable saddle drift detected (forward/back or sliding).")
        recs.append("Review girth placement/tension and pad grip. Round-barrel horses may need stability solutions.")

    if shoulder_std > align_threshold or hip_std > align_threshold:
        flags.append("Shoulder/hip alignment variance detected (uneven rider balance).")
        recs.append("Coach note: work on even weight in both stirrups; ride straight lines and check saddle straightness.")

    if bounce > bounce_threshold:
        flags.append("Higher vertical bounce detected in saddle midpoint.")
        recs.append("Focus on core engagement and softer landing in the saddle to reduce bounce.")

    if clearance_collapse:
        flags.append(f"Withers clearance risk (min {clearance_min:.1f}px below threshold).")
        recs.append("Check panel flocking/pad thickness to maintain withers clearance.")

    if bridging_proxy > 4.0:
        flags.append("Rear vs front clearance variation suggests bridging tendency.")
        recs.append("Evaluate panel contact; consider shim or refit to reduce bridging.")

    if cadence == 0.0:
        recs.append("Cadence not detected (short video or limited motion); film 30-60s with consistent trot/canter for rhythm.")
    elif cadence < 0.5:
        recs.append("Rhythm appears slow/irregular; aim for a steady tempo to improve stability.")

    saddle_stability = max(0.0, 100.0 - (rock * 6.0) - (drift_x * 0.5) - (abs(pitch) * 2.0) - ((shoulder_std + hip_std) * 0.5))
    saddle_stability = int(min(100.0, saddle_stability))

    rider_smoothness = max(0.0, 100.0 - metrics["rock_amplitude_deg"] * 8.0 - metrics["pitch_std_deg"] * 4.0)
    rider_consistency = max(0.0, 100.0 - (shoulder_std + hip_std) * 3.0)
    rider_control = max(0.0, 100.0 - drift_x * 0.6)
    rider_score = int(min(100.0, (rider_smoothness * 0.4 + rider_consistency * 0.3 + rider_control * 0.3)))

    rider_level = "Beginner"
    if rider_score >= 75:
        rider_level = "Advanced"
    elif rider_score >= 50:
        rider_level = "Intermediate"

    recs.append(f"Rider level estimate: {rider_level} ({rider_score}/100). Focus on steady shoulders/hips and straight lines.")

    fit_risk = "Low"
    if len(flags) == 1:
        fit_risk = "Medium"
    elif len(flags) >= 2:
        fit_risk = "High"

    stability_label = "Excellent" if saddle_stability >= 80 else "Good" if saddle_stability >= 55 else "Needs Attention"

    coach_good = []
    if rock < rock_threshold:
        coach_good.append("Stable rocking across strides.")
    if drift_x < drift_threshold:
        coach_good.append("Minimal forward/back drift observed.")
    if clearance_min and not clearance_collapse:
        coach_good.append("Withers clearance maintained through the ride.")

    coach_improve = []
    if bounce > bounce_threshold:
        coach_improve.append("Reduce vertical bounce by engaging core and following the motion.")
    if shoulder_std > align_threshold or hip_std > align_threshold:
        coach_improve.append("Equalize weight in both stirrups to level shoulders/hips.")
    if drift_x > drift_threshold:
        coach_improve.append("Improve saddle stability with tack adjustments or pad choice.")

    drills = [
        "Posting trot with light contact: focus on even shoulder height for 30s intervals.",
        "Two-point over ground poles to stabilize core and reduce bounce.",
        "Ride straight lines with visual markers to monitor drift and alignment.",
    ]

    return {
        "scores": {
            "saddle_stability": saddle_stability,
            "stability_label": stability_label,
            "fit_risk": fit_risk,
            "rider_score": rider_score,
            "rider_level": rider_level,
        },
        "tracking_confidence": tracking_conf,
        "flags": flags,
        "recommendations": recs,
        "coach": {
            "doing_well": coach_good,
            "to_improve": coach_improve,
            "drills": drills,
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
    pdf_html = render_pdf_report_html(analysis_id, meta, metrics, scored, mark_scores)
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


def render_report_html(
    analysis_id: str,
    meta: dict,
    metrics: Dict[str, Any],
    scored: Dict,
    stick_svg: str,
    growth_svg: str,
    mark_chart_svg: str,
    pdf_available: bool,
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
            <h1>Saddle Fit Analysis</h1>
            <p class="muted">Clear snapshot of stability, alignment, rider balance, and gear safety.</p>
          </div>
          <div style="text-align:right;">
            <div class="pill">ID: %%ANALYSIS_ID%%</div>
            <div class="muted" style="margin-top:4px;">Scheme: %%SCHEME%% | Saddle: %%SADDLE%%</div>
          </div>
        </div>

        <div class="summary">
          <div class="stat"><div class="k">Saddle stability</div><div class="v">%%SADDLE_STABILITY%%</div><div class="muted">%%STABILITY_LABEL%%</div></div>
          <div class="stat"><div class="k">Rider level</div><div class="v">%%RIDER_LEVEL%%</div><div class="muted">%%RIDER_SCORE%%</div></div>
          <div class="stat"><div class="k">Fit risk</div><div class="v">%%FIT_RISK%%</div><div class="muted">%%WARNINGS_COUNT%% warning(s)</div></div>
          <div class="stat"><div class="k">Tracking quality</div><div class="v">%%TRACK_PCT%%%</div><div class="muted">%%TRACK_FRAMES%% frames</div></div>
          <div class="stat"><div class="k">Gear & safety</div><div class="v">%%GEAR_STATUS%%</div><div class="muted">%%GEAR_USED_TEXT%%</div></div>
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
        "SADDLE": meta.get("saddle_type", ""),
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
            <h1>Saddle Fit Report</h1>
            <div class="muted">ID: {analysis_id} | Scheme: {meta.get("horse_scheme","")} | Saddle: {meta.get("saddle_type","")}</div>
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
    show_video: bool = True,
    include_analysis: bool = False,
    mark_scores_a: Optional[Dict[str, Dict[str, Any]]] = None,
    mark_scores_b: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    mark_scores_a = mark_scores_a or compute_mark_scores(metrics_a, scored_a)
    mark_scores_b = mark_scores_b or compute_mark_scores(metrics_b, scored_b)

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
              </div>
              <div>
                <span class="pill">ID: {compare_id}</span>
                <span class="pill">Scheme: {meta.get("horse_scheme","")}</span>
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
          <span class="pill">Scheme: {meta.get("horse_scheme","")}</span>
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
) -> Tuple[bool, str]:
    storage.ensure_runtime_directories()
    mark_scores_a = compute_mark_scores(metrics_a, scored_a)
    mark_scores_b = compute_mark_scores(metrics_b, scored_b)
    report_dir = storage.comparison_report_dir(compare_id)
    report_dir.mkdir(parents=True, exist_ok=True)

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
    scored = score_and_recommend(metrics, meta["horse_scheme"], meta["saddle_type"])
    scored["gear_assessment"] = gear_assessment
    scored["gear_detection"] = detection_data
    scored["gear_used"] = merged_gear
    mark_scores = compute_mark_scores(metrics, scored)

    result = {
        "analysis_id": analysis_id,
        "horse_scheme": meta["horse_scheme"],
        "saddle_type": meta["saddle_type"],
        "gear": meta.get("gear", {}),
        "gear_detected": detection_data,
        "gear_detection": detection_data,
        "gear_used": merged_gear,
        "video_filename": meta["video_filename"],
        "metrics": metrics,
        "scores": scored["scores"],
        "flags": scored["flags"],
        "recommendations": scored["recommendations"],
        "calibration_points": points,
        "auto_detect_confidence": auto["confidence"],
        "gear_assessment": gear_assessment,
        "marks": mark_scores,
    }

    with open(run_dir_path / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result, metrics, scored, auto["confidence"]


# ------------------- UI Pages -------------------
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
            <button class="ghost" type="button" onclick="openSchemeGuide()">Scheme guide</button>
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
            <p class="hint" style="margin-top:4px;">Designed for quick uploads and clean rider/horse metrics—no extra fluff.</p>
            <form id="uploadForm" action="/start" method="post" enctype="multipart/form-data">
              <label>Video (MP4 / MOV)</label>
              <input type="file" name="video" accept="video/*" required />

              <div class="input-row">
                <div>
                  <label>Horse type / discipline</label>
                  <select name="horse_scheme">
                    {schemes_select_html("high_wither")}
                  </select>
                  <div class="hint">
                    <button type="button" class="mini-btn" onclick="openSchemeGuide()">Scheme guide</button>
                    <span style="margin-left:6px;">Quick set:</span>
                    <div class="scheme-buttons">
                      <button type="button" class="mini-btn" onclick="quickScheme('trail')">Trail</button>
                      <button type="button" class="mini-btn" onclick="quickScheme('dressage')">Arena</button>
                      <button type="button" class="mini-btn" onclick="quickScheme('racing')">Speed</button>
                      <button type="button" class="mini-btn" onclick="quickScheme('show_jumping')">Jumping</button>
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
                <span class="tag">Shared scheme</span>
                <span class="tag">Progress callouts</span>
              </div>
            </div>
            <form id="compareForm" action="/compare_start" method="post" enctype="multipart/form-data">
              <label>Video A (MP4 / MOV)</label>
              <input type="file" name="video_a" accept="video/*" required />
              <label>Video B (MP4 / MOV)</label>
              <input type="file" name="video_b" accept="video/*" required />

              <div class="input-row">
                <div>
                  <label>Horse scheme / discipline</label>
                  <select name="horse_scheme_compare">
                    {schemes_select_html("high_wither")}
                  </select>
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
        function openSchemeGuide() {{
          const modal = document.getElementById("schemeModal");
          if (modal) modal.style.display = "flex";
        }}
        function closeSchemeGuide() {{
          const modal = document.getElementById("schemeModal");
          if (modal) modal.style.display = "none";
        }}
        function quickScheme(val) {{
          const selMain = document.querySelector("select[name='horse_scheme']");
          const selCompare = document.querySelector("select[name='horse_scheme_compare']");
          if (selMain) selMain.value = val;
          if (selCompare) selCompare.value = val;
        }}
      </script>
    </body>
    </html>
    """


if MULTIPART_AVAILABLE:
    @app.post("/start", response_class=HTMLResponse)
    async def start(
        video: UploadFile = File(...),
        horse_scheme: str = Form("high_wither"),
        saddle_type: str = Form("english"),
    ):
        if video is None or not video.filename:
            return HTMLResponse("<h3>No video uploaded.</h3>", status_code=400)

        storage.ensure_runtime_directories()
        analysis_id = str(uuid.uuid4())
        run_dir = storage.analysis_output_dir(analysis_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        filename = os.path.basename(video.filename) or "uploaded_video.mp4"
        video_path = run_dir / filename
        with open(video_path, "wb") as f:
            f.write(await video.read())

        frame_path = run_dir / "frame0.png"
        w, h = extract_first_frame(video_path, frame_path)

        meta = {
            "analysis_id": analysis_id,
            "video_filename": filename,
            "horse_scheme": horse_scheme,
            "saddle_type": saddle_type,
            "gear": {},
            "gear_detection": {},
            "gear_used": {},
            "frame_width": w,
            "frame_height": h,
        }
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        try:
            result, metrics, scored, confidence = analyze_video_auto(run_dir, analysis_id, meta)
        except Exception as exc:
            # If auto fails, allow manual calibration instead of blocking upload
            return HTMLResponse(f"""
            <html><body style="font-family: Arial; margin: 32px;">
              <h3>Auto-detection failed: {exc}</h3>
              <p>Please try a clearer side-view video, or switch to manual calibration.</p>
              <p><a href="/calibrate/{analysis_id}">Go to manual calibration for this video</a></p>
              <p><a href="/">Back to home</a></p>
            </body></html>
            """, status_code=400)

        points = result.get("calibration_points", {})
        pdf_generated, final_html = save_report_assets(run_dir, analysis_id, meta, metrics, scored, points)
        return HTMLResponse(final_html)


    @app.post("/compare_start", response_class=HTMLResponse)
    async def compare_start(
        video_a: UploadFile = File(...),
        video_b: UploadFile = File(...),
        horse_scheme_compare: str = Form("high_wither"),
        saddle_type_compare: str = Form("english"),
    ):
        if not video_a.filename or not video_b.filename:
            return HTMLResponse("<h3>Both videos are required.</h3>", status_code=400)

        storage.ensure_runtime_directories()
        compare_id = str(uuid.uuid4())
        base_dir = storage.comparison_output_dir(compare_id)
        run_a = storage.comparison_case_dir(compare_id, "a")
        run_b = storage.comparison_case_dir(compare_id, "b")
        run_a.mkdir(parents=True, exist_ok=True)
        run_b.mkdir(parents=True, exist_ok=True)

        # Save videos and frames
        fname_a = os.path.basename(video_a.filename) or "video_a.mp4"
        fname_b = os.path.basename(video_b.filename) or "video_b.mp4"
        path_a = run_a / fname_a
        path_b = run_b / fname_b
        with open(path_a, "wb") as f:
            f.write(await video_a.read())
        with open(path_b, "wb") as f:
            f.write(await video_b.read())

        wa, ha = extract_first_frame(path_a, run_a / "frame0.png")
        wb, hb = extract_first_frame(path_b, run_b / "frame0.png")

        meta_base = {
            "horse_scheme": horse_scheme_compare,
            "saddle_type": saddle_type_compare,
        }
        meta_a = {**meta_base, "analysis_id": compare_id + "_a", "video_filename": fname_a, "frame_width": wa, "frame_height": ha}
        meta_b = {**meta_base, "analysis_id": compare_id + "_b", "video_filename": fname_b, "frame_width": wb, "frame_height": hb}

        try:
            _, metrics_a, scored_a, conf_a = analyze_video_auto(run_a, meta_a["analysis_id"], meta_a)
            _, metrics_b, scored_b, conf_b = analyze_video_auto(run_b, meta_b["analysis_id"], meta_b)
        except Exception as exc:
            return HTMLResponse(f"<h3>Comparison failed:</h3><p>{exc}</p>", status_code=400)

        compare_meta = {"horse_scheme": horse_scheme_compare, "saddle_type": saddle_type_compare}
        pdf_generated, final_html = save_compare_report(base_dir, compare_id, compare_meta, fname_a, fname_b, metrics_a, metrics_b, scored_a, scored_b)

        with open(base_dir / "compare.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "compare_id": compare_id,
                    "video_a": {"file": fname_a, "confidence": conf_a, "metrics": metrics_a, "scores": scored_a["scores"]},
                    "video_b": {"file": fname_b, "confidence": conf_b, "metrics": metrics_b, "scores": scored_b["scores"]},
                },
                f,
                indent=2,
            )

        return HTMLResponse(final_html)
else:
    @app.post("/start", response_class=HTMLResponse)
    async def start_missing():
        return HTMLResponse("<h3>Server missing dependency python-multipart. Please install it with 'pip install python-multipart' and restart.</h3>", status_code=500)

    @app.post("/compare_start", response_class=HTMLResponse)
    async def compare_start_missing():
        return HTMLResponse("<h3>Server missing dependency python-multipart. Please install it with 'pip install python-multipart' and restart.</h3>", status_code=500)


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

    res = auto_calibrate_points(frame, meta["horse_scheme"], meta["saddle_type"])
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
    horse_scheme = meta["horse_scheme"]
    saddle_type = meta["saddle_type"]

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
            <div class="step"><b>Horse scheme:</b> {horse_scheme}</div>
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
    scored = score_and_recommend(metrics, meta["horse_scheme"], meta["saddle_type"])
    scored["gear_assessment"] = gear_assessment
    scored["gear_detection"] = detection_data
    scored["gear_used"] = merged_gear
    mark_scores = compute_mark_scores(metrics, scored)

    result = {
        "analysis_id": analysis_id,
        "horse_scheme": meta["horse_scheme"],
        "saddle_type": meta["saddle_type"],
        "gear": meta.get("gear", {}),
        "gear_detected": detection_data,
        "gear_detection": detection_data,
        "gear_used": merged_gear,
        "video_filename": meta["video_filename"],
        "metrics": metrics,
        "scores": scored["scores"],
        "flags": scored["flags"],
        "recommendations": scored["recommendations"],
        "calibration_points": init_points,
        "gear_assessment": gear_assessment,
        "marks": mark_scores,
    }

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
        return HTMLResponse("<h3>Report not found</h3>", status_code=404)
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
    path = first_existing_path(
        storage.comparison_report_dir(compare_id) / "report.html",
        storage.comparison_output_dir(compare_id) / "report.html",
        LEGACY_OUTPUTS_DIR / "compare" / compare_id / "report.html",
    )
    if path is None:
        return HTMLResponse("<h3>Comparison report not found</h3>", status_code=404)
    return path.read_text(encoding="utf-8")


@app.get("/compare_report/{compare_id}.pdf")
def compare_report_pdf(compare_id: str):
    pdf_path = first_existing_path(
        storage.comparison_report_dir(compare_id) / "report.pdf",
        storage.comparison_output_dir(compare_id) / "report.pdf",
        LEGACY_OUTPUTS_DIR / "compare" / compare_id / "report.pdf",
    )
    if pdf_path is not None:
        return FileResponse(os.fspath(pdf_path), media_type="application/pdf", filename=f"compare_{compare_id}.pdf")

    html_path = first_existing_path(
        storage.comparison_report_dir(compare_id) / "report.html",
        storage.comparison_output_dir(compare_id) / "report.html",
        LEGACY_OUTPUTS_DIR / "compare" / compare_id / "report.html",
    )
    if html_path is None:
        return HTMLResponse("<h3>Comparison PDF not found (install weasyprint to enable PDF export).</h3>", status_code=404)

    fallback_dir = storage.comparison_report_dir(compare_id)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fallback_dir / "report.pdf"
    try:
        pdf_renderer = get_weasyprint()
        if pdf_renderer is not None:
            pdf_renderer.HTML(filename=os.fspath(html_path), base_url=BASE_DIR).write_pdf(os.fspath(pdf_path))
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
