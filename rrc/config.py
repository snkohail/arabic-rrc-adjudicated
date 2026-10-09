"""Central configuration: role inventory, paths, and pre-declared parameters.

All methodological constants live here so that the paper can cite one place.
Nothing in this module reads data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

# --------------------------------------------------------------------------- roles
ROLES = [
    "PREAMBLE",
    "FACTS",
    "ISSUE",
    "ARGUMENT_PLAINTIFF",
    "ARGUMENT_DEFENDANT",
    "LAW_REFERENCE",
    "ANALYSIS",
    "DECISION",
    "DECISION_APPEAL",
]
ROLE_INDEX: Dict[str, int] = {r: i for i, r in enumerate(ROLES)}

# --------------------------------------------------------------------------- data authority
# Folder names inside the (private, never committed) annotation root.
ANNOTATOR_DIRS = {"A": "annotator_A", "B": "annotator_B", "C": "Annotator_C"}
ENV_ANNOTATION_ROOT = "RRC_ANNOTATION_ROOT"

# --------------------------------------------------------------------------- repo paths
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"      # committed: counts and hashes only, no text
SPLITS_DIR = REPO_ROOT / "splits"        # committed: frozen grid + split manifests (ids/offsets/hashes, no text)
CONFIG_DIR = REPO_ROOT / "configs"
LOCAL_CONFIG = CONFIG_DIR / "local.json"  # git-ignored: {"annotation_root": ..., "data_dir": ...}
ENV_DATA_DIR = "RRC_DATA_DIR"


def _local_cfg() -> dict:
    import json
    if LOCAL_CONFIG.exists():
        return json.loads(LOCAL_CONFIG.read_text())
    return {}


def data_dir() -> Path:
    """Private, never-committed location for derived datasets and caches (contain judgment text)."""
    val = os.environ.get(ENV_DATA_DIR) or _local_cfg().get("data_dir") or str(REPO_ROOT / "data")
    return Path(val).expanduser().resolve()


DATA_DIR = data_dir()
DERIVED_DIR = DATA_DIR / "derived"
CACHE_DIR = DATA_DIR / "cache"


def annotation_root(arg: Optional[str] = None) -> Path:
    """Resolve the private annotation root: CLI arg > $RRC_ANNOTATION_ROOT > configs/local.json."""
    val = arg or os.environ.get(ENV_ANNOTATION_ROOT) or _local_cfg().get("annotation_root")
    if not val:
        raise SystemExit(
            f"Annotation root not given. Pass --annotation-root, set ${ENV_ANNOTATION_ROOT}, or create {LOCAL_CONFIG}."
        )
    p = Path(val).expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(f"Annotation root does not exist: {p}")
    return p


# --------------------------------------------------------------------------- modeling corpus
# Exact-duplicate judgment texts: both copies stay in the corpus provenance/audit, one canonical copy is
# used for modeling (pre-declared, 2026-09-02). The retained copy is the lower canonical id.
EXCLUDED_FROM_MODELING = {
    "C0047": "exact duplicate text of C0019 (identical sha256); C0019 retained as the canonical copy",
}


# --------------------------------------------------------------------------- pre-declared parameters
SPLIT = {
    "name": "split_v1",
    "seed": 13,
    "sizes": {"train": 139, "val": 30, "test": 30},   # 199 unique-text judgments
    # stratification indicators (split attributes only; never model inputs):
    #   9 role-presence flags, MULTI_ROLE_DOC, and a coarse regions-per-judgment tercile bin
    "volume_bins": ["VOLUME_LOW", "VOLUME_MED", "VOLUME_HIGH"],
}

# Duplicate / related-judgment grouping (text only, computed before splitting).
DEDUP = {
    "shingle_words": 8,
    "jaccard_threshold": 0.20,
    "containment_threshold": 0.50,
}

# Encoder budget used for the length audit (512 minus [CLS]/[SEP]).
ENCODER_INPUT_LIMIT = 510
TOKENIZER_ARABERT = "aubmindlab/bert-base-arabertv2"        # Farasa-segmented pre-training
TOKENIZER_ARABERT_V02 = "aubmindlab/bert-base-arabertv02"  # raw-text pre-training
TOKENIZER_CAMELBERT = "CAMeL-Lab/bert-base-arabic-camelbert-msa"

# Sentence grid used for S2 (see rrc/sentences.py for the rule); frozen in splits/grid_v1.json.
GRID_VERSION = "grid_v1"
# Projection of C onto the grid: a sentence whose maximum-overlap role is not unique is marked
# AMBIGUOUS_PROJECTION and excluded from S2 (no artificial tie-breaking); a sentence overlapping no
# adjudicated span is UNLABELED and excluded (no NONE class). Pre-declared, 2026-09-02.
PROJECTION_STATUSES = ("LABELED", "AMBIGUOUS_PROJECTION", "UNLABELED")

# Minimal C normalisation applied at load time (see rrc/normalize.py). Pre-declared, 2026-09-02:
# only mechanical artefacts are repaired; inherited/redundant labels are kept as adjudicated; all 200
# judgments are kept, including the exact-duplicate pair C0019/C0047 (grouped in one split).
C_NORMALIZATION = {"version": "cnorm_v1", "min_span_chars": 5, "near_identical_max_len_diff": 20}

# Provenance vocabulary found in C files (see rrc/validate.py).
UNREVIEWED_PROVENANCE_PREFIXES = ("unadjudicated",)
UNION_CANDIDATE_MARKERS = {"UNION"}
# The per-row ``provenance`` / ``candidate_source`` fields and the file-level ``merge`` block in the
# C files are artefacts of the annotation tool's earlier machine pass and do NOT describe the final
# human adjudication.  They are counted for transparency but are not
# used as evidence about adjudication status.
PROVENANCE_FLAGS_ARE_TOOL_ARTEFACTS = True
C_ADJUDICATION_ATTESTATION = ("Annotator_C is treated as the human-adjudicated final gold; every multi-role region "
                              "is taken as an explicit adjudicator decision. Metadata flags left by the annotation "
                              "tool are not used as evidence.")
