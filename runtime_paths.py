import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
IS_VERCEL = bool(os.getenv("VERCEL"))

# Vercel's deployment filesystem is read-only at runtime. Keep mutable files
# under /tmp there and under a local runtime/ directory during development.
if IS_VERCEL:
    RUNTIME_ROOT = Path(tempfile.gettempdir()) / "saddlefitsystem"
else:
    RUNTIME_ROOT = PROJECT_ROOT / "runtime"

UPLOAD_DIR = RUNTIME_ROOT / "uploads"
OUTPUT_DIR = RUNTIME_ROOT / "outputs"
COMPARE_DIR = OUTPUT_DIR / "compare"
REPORT_DIR = RUNTIME_ROOT / "reports"
TEMP_DIR = RUNTIME_ROOT / "temp"


def ensure_runtime_directories() -> None:
    for directory in [
        RUNTIME_ROOT,
        UPLOAD_DIR,
        OUTPUT_DIR,
        COMPARE_DIR,
        REPORT_DIR,
        TEMP_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def analysis_output_dir(analysis_id: str) -> Path:
    return OUTPUT_DIR / analysis_id


def analysis_report_dir(analysis_id: str) -> Path:
    return REPORT_DIR / analysis_id


def comparison_output_dir(compare_id: str) -> Path:
    return COMPARE_DIR / compare_id


def comparison_case_dir(compare_id: str, label: str) -> Path:
    return comparison_output_dir(compare_id) / label


def comparison_report_dir(compare_id: str) -> Path:
    return REPORT_DIR / "compare" / compare_id
