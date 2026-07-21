from __future__ import annotations

from typing import Dict, List, Tuple


HORSE_PROFILE_OPTIONS: List[Tuple[str, str]] = [
    ("high_wither", "High-wither (prominent withers)"),
    ("round_barrel", "Round-barrel (broad ribcage)"),
    ("narrow_build", "Narrow build (slimmer frame)"),
    ("wide_build", "Wide build (broad back)"),
    ("short_back", "Short back (compact support area)"),
    ("long_back", "Long back (extended support area)"),
]

DISCIPLINE_OPTIONS: List[Tuple[str, str]] = [
    ("general_riding", "General Riding"),
    ("trail_riding", "Trail Riding"),
    ("arena_riding", "Arena Riding"),
    ("dressage", "Dressage"),
    ("show_jumping", "Show Jumping"),
    ("eventing", "Eventing"),
    ("hunter", "Hunter"),
    ("equitation", "Equitation"),
    ("endurance_riding", "Endurance Riding"),
    ("western_pleasure", "Western Pleasure"),
    ("reining", "Reining"),
    ("barrel_racing", "Barrel Racing"),
    ("cutting", "Cutting"),
    ("polo", "Polo"),
    ("racing_gallop", "Racing / Gallop Analysis"),
]

HORSE_PROFILE_CONFIGS: Dict[str, Dict[str, float | str]] = {
    "high_wither": {
        "label": "High-wither",
        "front_down_threshold": -2.5,
        "rock_threshold_adjust": 0.0,
        "drift_threshold_adjust": 0.0,
        "clearance_threshold_adjust": 4.0,
        "notes": "Prioritize withers clearance and front balance.",
    },
    "round_barrel": {
        "label": "Round-barrel",
        "front_down_threshold": -4.5,
        "rock_threshold_adjust": 1.5,
        "drift_threshold_adjust": 8.0,
        "clearance_threshold_adjust": 0.0,
        "notes": "Prioritize girth security and lateral stability.",
    },
    "narrow_build": {
        "label": "Narrow build",
        "front_down_threshold": -3.5,
        "rock_threshold_adjust": 0.0,
        "drift_threshold_adjust": -2.0,
        "clearance_threshold_adjust": 2.0,
        "notes": "Watch for side-to-side movement and panel contact.",
    },
    "wide_build": {
        "label": "Wide build",
        "front_down_threshold": -4.0,
        "rock_threshold_adjust": 1.0,
        "drift_threshold_adjust": 4.0,
        "clearance_threshold_adjust": -1.0,
        "notes": "Watch for bridging, front roll, and support length.",
    },
    "short_back": {
        "label": "Short back",
        "front_down_threshold": -3.5,
        "rock_threshold_adjust": 1.0,
        "drift_threshold_adjust": 0.0,
        "clearance_threshold_adjust": 1.0,
        "notes": "Favor compact panels and avoid excessive rear pressure.",
    },
    "long_back": {
        "label": "Long back",
        "front_down_threshold": -4.0,
        "rock_threshold_adjust": 0.0,
        "drift_threshold_adjust": 0.0,
        "clearance_threshold_adjust": 0.0,
        "notes": "Check long-area contact and even support distribution.",
    },
}

DISCIPLINE_CONFIGS: Dict[str, Dict[str, object]] = {
    "general_riding": {
        "label": "General Riding",
        "focus": ["balance", "posture", "steady rhythm", "safe tack"],
        "priority_metrics": [
            "Rider Balance",
            "Rider Posture",
            "Horse Rhythm",
            "Saddle Stability",
            "Symmetry",
        ],
        "weights": {"rider": 0.35, "horse": 0.20, "saddle": 0.25, "symmetry": 0.20},
        "expected_rhythm_hz": (0.9, 1.8),
        "expected_motion": "steady",
        "notes": "Balanced seat, steady contact, and clean straightness matter most.",
    },
    "trail_riding": {
        "label": "Trail Riding",
        "focus": ["long-duration stability", "calm rhythm", "reduced drift"],
        "priority_metrics": [
            "Saddle Stability",
            "Horse Consistency",
            "Rider Stability",
            "Rider Balance",
        ],
        "weights": {"rider": 0.30, "horse": 0.25, "saddle": 0.30, "symmetry": 0.15},
        "expected_rhythm_hz": (0.7, 1.6),
        "expected_motion": "steady",
        "notes": "Longer, steadier work; prioritize comfort and consistency.",
    },
    "arena_riding": {
        "label": "Arena Riding",
        "focus": ["posture", "straightness", "consistent rhythm"],
        "priority_metrics": [
            "Rider Posture",
            "Rider Symmetry",
            "Horse Rhythm",
            "Saddle Position",
        ],
        "weights": {"rider": 0.33, "horse": 0.20, "saddle": 0.27, "symmetry": 0.20},
        "expected_rhythm_hz": (0.9, 2.0),
        "expected_motion": "steady",
        "notes": "Use the arena to emphasize straight lines and repeatable balance.",
    },
    "dressage": {
        "label": "Dressage",
        "focus": ["symmetry", "seat stability", "posture", "straightness"],
        "priority_metrics": [
            "Rider Symmetry",
            "Seat Stability",
            "Posture",
            "Horse Symmetry",
            "Horse Rhythm",
        ],
        "weights": {"rider": 0.38, "horse": 0.18, "saddle": 0.22, "symmetry": 0.22},
        "expected_rhythm_hz": (1.0, 2.2),
        "expected_motion": "collected",
        "notes": "Precision and symmetry matter more than speed.",
    },
    "show_jumping": {
        "label": "Show Jumping",
        "focus": ["lower-leg stability", "two-point balance", "hip/knee angles"],
        "priority_metrics": [
            "Lower-Leg Stability",
            "Hip Angle",
            "Knee Angle",
            "Rider Balance",
            "Saddle Stability",
        ],
        "weights": {"rider": 0.40, "horse": 0.16, "saddle": 0.18, "symmetry": 0.26},
        "expected_rhythm_hz": (1.1, 2.4),
        "expected_motion": "forward",
        "notes": "Approach balance and leg security matter more than topline metrics alone.",
    },
    "eventing": {
        "label": "Eventing",
        "focus": ["balance", "adaptability", "steady rider support"],
        "priority_metrics": [
            "Rider Balance",
            "Horse Movement",
            "Saddle Stability",
            "Symmetry",
        ],
        "weights": {"rider": 0.34, "horse": 0.22, "saddle": 0.22, "symmetry": 0.22},
        "expected_rhythm_hz": (1.0, 2.2),
        "expected_motion": "forward",
        "notes": "Blend the demands of jumping and flatwork.",
    },
    "hunter": {
        "label": "Hunter",
        "focus": ["quiet rider", "stable approach", "smooth rhythm"],
        "priority_metrics": [
            "Rider Stability",
            "Horse Rhythm",
            "Horse Consistency",
            "Saddle Position",
        ],
        "weights": {"rider": 0.34, "horse": 0.22, "saddle": 0.22, "symmetry": 0.22},
        "expected_rhythm_hz": (1.0, 2.0),
        "expected_motion": "smooth",
        "notes": "Presentation and smoothness matter more than aggressive motion.",
    },
    "equitation": {
        "label": "Equitation",
        "focus": ["rider position", "balance", "leg alignment"],
        "priority_metrics": [
            "Rider Posture",
            "Rider Leg Position",
            "Rider Symmetry",
            "Rider Balance",
        ],
        "weights": {"rider": 0.42, "horse": 0.16, "saddle": 0.18, "symmetry": 0.24},
        "expected_rhythm_hz": (0.9, 2.0),
        "expected_motion": "balanced",
        "notes": "Rider form is emphasized strongly.",
    },
    "endurance_riding": {
        "label": "Endurance Riding",
        "focus": ["sustained stability", "horse consistency", "comfort"],
        "priority_metrics": [
            "Horse Consistency",
            "Saddle Stability",
            "Rider Stability",
            "Rhythm",
        ],
        "weights": {"rider": 0.28, "horse": 0.30, "saddle": 0.27, "symmetry": 0.15},
        "expected_rhythm_hz": (0.7, 1.6),
        "expected_motion": "steady",
        "notes": "Endurance favors low fatigue and conservative saddle movement.",
    },
    "western_pleasure": {
        "label": "Western Pleasure",
        "focus": ["soft motion", "calm posture", "quiet hands/seat"],
        "priority_metrics": [
            "Horse Rhythm",
            "Rider Posture",
            "Rider Stability",
            "Saddle Balance",
        ],
        "weights": {"rider": 0.31, "horse": 0.27, "saddle": 0.22, "symmetry": 0.20},
        "expected_rhythm_hz": (0.8, 1.8),
        "expected_motion": "soft",
        "notes": "Motion should look smooth and unhurried.",
    },
    "reining": {
        "label": "Reining",
        "focus": ["seat stability", "lateral control", "quiet legs"],
        "priority_metrics": [
            "Rider Stability",
            "Rider Leg Position",
            "Horse Consistency",
            "Saddle Balance",
        ],
        "weights": {"rider": 0.36, "horse": 0.22, "saddle": 0.20, "symmetry": 0.22},
        "expected_rhythm_hz": (0.8, 1.8),
        "expected_motion": "collected",
        "notes": "Lateral stability matters more than straight-line speed.",
    },
    "barrel_racing": {
        "label": "Barrel Racing",
        "focus": ["dynamic balance", "turn entry stability", "acceleration control"],
        "priority_metrics": [
            "Dynamic Balance",
            "Saddle Stability",
            "Horse Movement",
            "Rider Symmetry",
        ],
        "weights": {"rider": 0.38, "horse": 0.22, "saddle": 0.20, "symmetry": 0.20},
        "expected_rhythm_hz": (1.1, 2.5),
        "expected_motion": "dynamic",
        "notes": "Use motion consistency and rider balance proxies from the available camera view.",
    },
    "cutting": {
        "label": "Cutting",
        "focus": ["horse agility", "rider quietness", "balance under quick motion"],
        "priority_metrics": [
            "Horse Consistency",
            "Rider Stability",
            "Rider Balance",
            "Saddle Stability",
        ],
        "weights": {"rider": 0.34, "horse": 0.26, "saddle": 0.20, "symmetry": 0.20},
        "expected_rhythm_hz": (1.0, 2.2),
        "expected_motion": "dynamic",
        "notes": "Quick changes are expected, so assess stability across those changes.",
    },
    "polo": {
        "label": "Polo",
        "focus": ["upper-body stability", "dynamic balance", "lateral control"],
        "priority_metrics": [
            "Rider Stability",
            "Rider Balance",
            "Horse Movement",
            "Saddle Stability",
        ],
        "weights": {"rider": 0.38, "horse": 0.24, "saddle": 0.18, "symmetry": 0.20},
        "expected_rhythm_hz": (1.0, 2.4),
        "expected_motion": "dynamic",
        "notes": "High-speed motion and rotation make balance proxies more important.",
    },
    "racing_gallop": {
        "label": "Racing / Gallop Analysis",
        "focus": ["forward motion", "saddle stability", "rhythm"],
        "priority_metrics": [
            "Horse Rhythm",
            "Saddle Stability",
            "Rider Stability",
            "Horse Movement",
        ],
        "weights": {"rider": 0.30, "horse": 0.32, "saddle": 0.24, "symmetry": 0.14},
        "expected_rhythm_hz": (1.6, 3.6),
        "expected_motion": "forward",
        "notes": "Use the camera view to assess stability, cadence, and support under speed.",
    },
}

LEGACY_DISCIPLINE_ALIASES: Dict[str, str] = {
    "trail": "trail_riding",
    "dressage": "dressage",
    "show_jumping": "show_jumping",
    "eventing": "eventing",
    "hunter": "hunter",
    "equitation": "equitation",
    "endurance": "endurance_riding",
    "western": "western_pleasure",
    "western_pleasure": "western_pleasure",
    "reining": "reining",
    "barrel": "barrel_racing",
    "barrel_racing": "barrel_racing",
    "cutting": "cutting",
    "polo": "polo",
    "racing": "racing_gallop",
    "racing_gallop": "racing_gallop",
    "arena": "arena_riding",
    "arena_riding": "arena_riding",
    "general": "general_riding",
    "general_riding": "general_riding",
}

COMPARISON_SIGNIFICANCE_THRESHOLD = 3.0

PROFILE_KEYS = {key for key, _ in HORSE_PROFILE_OPTIONS}
DISCIPLINE_KEYS = {key for key, _ in DISCIPLINE_OPTIONS}


def normalize_horse_profile(value: str | None) -> str:
    if not value:
        return "high_wither"
    value = value.strip().lower()
    return value if value in PROFILE_KEYS else "high_wither"


def normalize_discipline(value: str | None) -> str:
    if not value:
        return "general_riding"
    value = value.strip().lower()
    value = LEGACY_DISCIPLINE_ALIASES.get(value, value)
    return value if value in DISCIPLINE_KEYS else "general_riding"


def split_legacy_scheme(value: str | None) -> Tuple[str, str]:
    normalized = (value or "").strip().lower()
    if normalized in PROFILE_KEYS:
        return normalized, "general_riding"
    discipline = normalize_discipline(normalized)
    if discipline != "general_riding":
        return "high_wither", discipline
    return "high_wither", "general_riding"


def get_profile_config(value: str | None) -> Dict[str, object]:
    return dict(HORSE_PROFILE_CONFIGS.get(normalize_horse_profile(value), HORSE_PROFILE_CONFIGS["high_wither"]))


def get_discipline_config(value: str | None) -> Dict[str, object]:
    return dict(DISCIPLINE_CONFIGS.get(normalize_discipline(value), DISCIPLINE_CONFIGS["general_riding"]))


def score_band(score: float | None) -> str:
    if score is None:
        return "Insufficient Data"
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Needs Attention"
    return "Potential Concern"


def confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "Insufficient Data"
    if confidence >= 80:
        return "Excellent"
    if confidence >= 65:
        return "Good"
    if confidence >= 45:
        return "Needs Attention"
    return "Potential Concern"


def evidence_status(available: bool, estimated: bool = False) -> str:
    if not available:
        return "Insufficient Data"
    return "Estimated" if estimated else "Measured"
