from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitFractions:
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 <= self.val < 1.0 and 0.0 <= self.test < 1.0):
            raise ValueError("val/test must be in [0, 1).")
        if self.val + self.test >= 1.0:
            raise ValueError("val + test must be < 1.")


def assign_patient_splits(
    manifest: list[dict[str, str]],
    *,
    seed: int = 42,
    fractions: SplitFractions = SplitFractions(),
    label_col: str = "label",
    patient_col: str = "patient_id",
) -> list[dict[str, str]]:
    """
    Assign train/val/test splits at the *patient* level to prevent leakage.

    Strategy (simple + deterministic):
    - group patients by label majority
    - split patients within each label bucket to preserve class balance approximately
    """
    rows = [dict(r) for r in manifest]
    if not rows:
        raise ValueError("Manifest is empty.")
    if patient_col not in rows[0] or label_col not in rows[0]:
        raise ValueError(f"Manifest must include {patient_col=} and {label_col=}")

    for r in rows:
        if r.get(label_col, "") == "":
            raise ValueError("Manifest contains empty labels; normalize labels before splitting.")
        if r.get(patient_col, "") == "":
            raise ValueError("Manifest contains empty patient_id values.")

    # patient -> label (require consistent label per patient)
    patient_to_label: dict[str, str] = {}
    for r in rows:
        pid = r[patient_col]
        lab = str(r[label_col])
        if pid in patient_to_label and patient_to_label[pid] != lab:
            raise ValueError(f"Inconsistent labels for patient {pid}: {patient_to_label[pid]} vs {lab}")
        patient_to_label[pid] = lab

    rng = random.Random(seed)
    patients = list(patient_to_label.keys())
    # Shuffle deterministically, then bucket by label for approximate stratification.
    rng.shuffle(patients)

    # split within each label bucket
    split_map: dict[str, SplitName] = {}
    label_to_patients: dict[str, list[str]] = {}
    for pid in patients:
        label_to_patients.setdefault(patient_to_label[pid], []).append(pid)

    for _label, bucket in label_to_patients.items():
        n = len(bucket)
        n_test = int(round(n * fractions.test))
        n_val = int(round(n * fractions.val))
        # Ensure we don't over-allocate due to rounding.
        n_test = min(n_test, n)
        n_val = min(n_val, n - n_test)
        n_train = n - n_val - n_test

        for pid in bucket[:n_train]:
            split_map[pid] = "train"
        for pid in bucket[n_train : n_train + n_val]:
            split_map[pid] = "val"
        for pid in bucket[n_train + n_val :]:
            split_map[pid] = "test"

    for r in rows:
        r["split"] = split_map.get(r[patient_col], "")
        if r["split"] == "":
            raise RuntimeError(f"Failed to assign split for patient: {r[patient_col]}")
    return rows


def assert_no_patient_leakage(rows: list[dict[str, str]], patient_col: str = "patient_id") -> None:
    """
    Assert each patient appears in exactly one split.
    """
    if not rows:
        raise ValueError("Manifest is empty.")
    if "split" not in rows[0]:
        raise ValueError("Manifest must include split column.")

    seen: dict[str, set[str]] = {}
    for r in rows:
        pid = r.get(patient_col, "")
        sp = r.get("split", "")
        if pid == "" or sp == "":
            raise ValueError("Missing patient_id or split in manifest rows.")
        seen.setdefault(pid, set()).add(sp)

    leaked = {pid: s for pid, s in seen.items() if len(s) > 1}
    if leaked:
        bad = list(leaked.keys())[:10]
        raise AssertionError(f"Patient leakage detected for {len(leaked)} patients, e.g. {bad}")


