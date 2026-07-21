from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    filename: str = ""
    mime_type: str = ""
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frames: Optional[int] = None


class AnalysisRequest(BaseModel):
    horse_profile: str = "high_wither"
    saddle_type: str = "english"
    discipline: str = "general_riding"


class ComparisonRequest(BaseModel):
    horse_profile: str = "high_wither"
    saddle_type: str = "english"
    discipline: str = "general_riding"


class MetricEntry(BaseModel):
    name: str
    value: Optional[float] = None
    display: str
    unit: str = ""
    status: str = "Insufficient Data"
    source: str = "estimated"
    note: str = ""


class ScoreBlock(BaseModel):
    overall: Optional[float] = None
    rider: Optional[float] = None
    horse_movement: Optional[float] = None
    saddle_stability: Optional[float] = None
    symmetry: Optional[float] = None
    discipline: Optional[float] = None
    rider_score: Optional[float] = None
    horse_score: Optional[float] = None
    saddle_score: Optional[float] = None
    rider_posture: Optional[float] = None
    rider_balance: Optional[float] = None
    rider_symmetry: Optional[float] = None
    rider_stability: Optional[float] = None
    rider_leg_position: Optional[float] = None
    horse_topline: Optional[float] = None
    horse_rhythm: Optional[float] = None
    horse_consistency: Optional[float] = None
    horse_symmetry: Optional[float] = None
    saddle_position: Optional[float] = None
    saddle_balance: Optional[float] = None
    fit_risk: str = ""
    rider_level: str = ""
    stability_label: str = ""
    labels: Dict[str, str] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    analysis_id: str
    created_at: str
    video_metadata: VideoMetadata = Field(default_factory=VideoMetadata)
    horse_profile: str
    saddle_type: str
    discipline: str
    confidence: float
    scores: ScoreBlock
    rider_metrics: List[MetricEntry] = Field(default_factory=list)
    horse_metrics: List[MetricEntry] = Field(default_factory=list)
    saddle_metrics: List[MetricEntry] = Field(default_factory=list)
    discipline_metrics: List[MetricEntry] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    visual_evidence: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    pose_summary: Dict[str, Any] = Field(default_factory=dict)
    summary_cards: Dict[str, Any] = Field(default_factory=dict)
    mark_scores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    gear_assessment: Dict[str, Any] = Field(default_factory=dict)
    gear_detection: Dict[str, Any] = Field(default_factory=dict)
    gear_used: Dict[str, Any] = Field(default_factory=dict)
    horse_profile_label: str = ""
    discipline_label: str = ""
    discipline_focus: List[str] = Field(default_factory=list)
    discipline_notes: str = ""
    calibration_points: Dict[str, Any] = Field(default_factory=dict)
    points: Dict[str, Any] = Field(default_factory=dict)
    scores_alias: Dict[str, Any] = Field(default_factory=dict)
    disclaimer: str
    report_html: Optional[str] = None
    report_url: Optional[str] = None
    pdf_url: Optional[str] = None


class AnalysisReportResponse(BaseModel):
    analysis: AnalysisResponse
    report_html: str
    report_url: Optional[str] = None
    pdf_url: Optional[str] = None


class ComparisonSide(BaseModel):
    analysis_id: str
    video_metadata: VideoMetadata = Field(default_factory=VideoMetadata)
    horse_profile: str
    saddle_type: str
    discipline: str
    confidence: float
    scores: ScoreBlock
    key_metrics: List[MetricEntry] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    metric: str
    ride_a: Optional[float] = None
    ride_b: Optional[float] = None
    delta: Optional[float] = None
    percent_change: Optional[float] = None
    direction: str = "No Significant Change"
    note: str = ""


class ComparisonResponse(BaseModel):
    comparison_id: str
    created_at: str
    horse_profile: str
    saddle_type: str
    discipline: str
    overall_summary: str
    ride_a: ComparisonSide
    ride_b: ComparisonSide
    comparisons: List[ComparisonRow] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    visual_evidence: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    label_a: str = ""
    label_b: str = ""
    analysis_a: Dict[str, Any] = Field(default_factory=dict)
    analysis_b: Dict[str, Any] = Field(default_factory=dict)
    comparison_rows_table: List[Dict[str, Any]] = Field(default_factory=list)
    stable_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    improved_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    declined_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    comparison_sections_html: str = ""
    disclaimer: str
    report_html: Optional[str] = None
    report_url: Optional[str] = None
    pdf_url: Optional[str] = None


class ComparisonReportResponse(BaseModel):
    comparison: ComparisonResponse
    report_html: str
    report_url: Optional[str] = None
    pdf_url: Optional[str] = None
