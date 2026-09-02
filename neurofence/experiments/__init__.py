"""Controlled experiments with known ground truth, for validating the detector."""

from .clean_model import CleanModel, prepare_clean_copy
from .controlled_backdoor import BackdoorGroundTruth, plant_controlled_backdoor
from .validation import ArmResult, ValidationReport, build_report

__all__ = [
    "ArmResult",
    "BackdoorGroundTruth",
    "CleanModel",
    "ValidationReport",
    "build_report",
    "plant_controlled_backdoor",
    "prepare_clean_copy",
]
