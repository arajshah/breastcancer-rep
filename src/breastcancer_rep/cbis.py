from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCbisId:
    """
    Parsed view-level identifier used by CBIS-DDSM.

    Example (from TCIA docs):
      Mass-Test_P_00038_LEFT_CC_1
    """

    modality: str  # "MASS" or "CALC" or other
    cohort: str  # "TRAINING" or "TEST" or other
    participant_id: str  # "00038"
    laterality: str  # "LEFT"/"RIGHT"/"UNKNOWN"
    view: str  # "CC"/"MLO"/"UNKNOWN"
    view_index: str  # e.g. "1" or "" if absent


_CBIS_PATIENT_ID_RE = re.compile(
    r"^(?P<modality>Mass|Calc)-(?P<cohort>Training|Test)_P_(?P<pid>\d+)"
    r"_(?P<lat>LEFT|RIGHT)_(?P<view>CC|MLO)(?:_(?P<idx>\d+))?$",
    flags=re.IGNORECASE,
)


def parse_cbis_patient_id(patient_id: str) -> ParsedCbisId:
    """
    Parse CBIS-DDSM patient_id strings (view-level IDs).

    If parsing fails, returns UNKNOWN laterality/view and preserves raw prefix info when possible.
    """
    s = (patient_id or "").strip()
    m = _CBIS_PATIENT_ID_RE.match(s)
    if not m:
        # Try to at least guess modality/cohort from prefix tokens.
        modality = "UNKNOWN"
        cohort = "UNKNOWN"
        if s.lower().startswith("mass-"):
            modality = "MASS"
        elif s.lower().startswith("calc-"):
            modality = "CALC"
        if "-training" in s.lower():
            cohort = "TRAINING"
        elif "-test" in s.lower():
            cohort = "TEST"
        return ParsedCbisId(
            modality=modality,
            cohort=cohort,
            participant_id="",
            laterality="UNKNOWN",
            view="UNKNOWN",
            view_index="",
        )

    modality = m.group("modality").upper()
    cohort = m.group("cohort").upper()
    participant_id = m.group("pid")
    laterality = m.group("lat").upper()
    view = m.group("view").upper()
    idx = m.group("idx") or ""
    return ParsedCbisId(
        modality=modality,
        cohort=cohort,
        participant_id=participant_id,
        laterality=laterality,
        view=view,
        view_index=idx,
    )


