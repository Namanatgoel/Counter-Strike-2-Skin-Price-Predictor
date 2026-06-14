"""
model_registry.py - Model persistence helpers
=============================================

Provides a small persistence layer for saving and loading trained model
bundles, along with lightweight fingerprints for detecting when historical
data has changed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ARTIFACT_DIR = Path("output") / "model_store"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "cs2_model_bundle.joblib"
MODEL_MANIFEST_PATH = ARTIFACT_DIR / "cs2_model_bundle.json"
ARTIFACT_VERSION = 1


def ensure_artifact_dir() -> None:
    """Create the artifact directory if it does not already exist."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def fingerprint_bytes(payload: bytes) -> str:
    """Return a stable SHA-256 fingerprint for the provided bytes."""
    return hashlib.sha256(payload).hexdigest()


def fingerprint_file(file_path: str | Path) -> str:
    """Return a stable SHA-256 fingerprint for a file on disk."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dataframe(df: pd.DataFrame | None) -> str:
    """Return a stable fingerprint for a DataFrame's values, index, and schema."""
    if df is None:
        return "none"

    if df.empty:
        return "empty"

    normalized = df.copy()
    normalized = normalized.sort_index(axis=0)
    normalized = normalized.sort_index(axis=1)

    values_hash = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    schema_payload = "|".join(map(str, normalized.columns)).encode("utf-8")
    dtype_payload = "|".join(map(str, normalized.dtypes)).encode("utf-8")

    return fingerprint_bytes(values_hash + schema_payload + dtype_payload)


def build_source_fingerprint(
    data_file_path: str | Path,
    metadata_df: pd.DataFrame | None,
    horizons: list[int],
) -> str:
    """Combine the data file, metadata, and forecast horizon into one fingerprint."""
    payload = "::".join([
        fingerprint_file(data_file_path),
        fingerprint_dataframe(metadata_df),
        ",".join(map(str, horizons)),
        str(ARTIFACT_VERSION),
    ])
    return fingerprint_bytes(payload.encode("utf-8"))


def bundle_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    """Strip large estimator objects down to a JSON-friendly summary."""
    return {
        "artifact_version": bundle.get("artifact_version", ARTIFACT_VERSION),
        "trained_at": bundle.get("trained_at"),
        "source_fingerprint": bundle.get("source_fingerprint"),
        "horizons": bundle.get("horizons", []),
        "feature_columns": bundle.get("feature_columns", []),
        "categorical_columns": bundle.get("categorical_columns", []),
        "best_params": bundle.get("best_params", {}),
        "evaluation": bundle.get("evaluation", {}),
    }


def save_model_bundle(
    bundle: dict[str, Any],
    bundle_path: str | Path = MODEL_BUNDLE_PATH,
    manifest_path: str | Path = MODEL_MANIFEST_PATH,
) -> None:
    """Persist a trained model bundle and a lightweight JSON manifest."""
    ensure_artifact_dir()
    joblib.dump(bundle, bundle_path)
    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(bundle_manifest(bundle), file_handle, indent=2, sort_keys=True, default=str)


def load_model_bundle(bundle_path: str | Path = MODEL_BUNDLE_PATH) -> dict[str, Any] | None:
    """Load a previously saved model bundle if it exists."""
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        return None
    return joblib.load(bundle_path)


def bundle_metric(bundle: dict[str, Any] | None, metric_name: str) -> float:
    """Return a stored metric for a bundle, or positive infinity when missing."""
    if not bundle:
        return float("inf")

    evaluation = bundle.get("evaluation", {})
    metric_value = evaluation.get(metric_name)
    if metric_value is None:
        return float("inf")
    return float(metric_value)


def bundle_is_current(
    bundle: dict[str, Any] | None,
    source_fingerprint: str,
    retrain_interval_days: int = 7,
) -> bool:
    """Return True when a bundle matches the current source data and is still fresh."""
    if not bundle:
        return False

    if bundle.get("source_fingerprint") != source_fingerprint:
        return False

    trained_at = bundle.get("trained_at")
    if not trained_at:
        return False

    try:
        trained_dt = datetime.fromisoformat(trained_at)
        if trained_dt.tzinfo is None:
            trained_dt = trained_dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - trained_dt.astimezone(timezone.utc)).total_seconds() / 86400
    except (TypeError, ValueError):
        return False

    return age_days < retrain_interval_days


def is_better_holdout(candidate_metrics: dict[str, Any], incumbent_metrics: dict[str, Any] | None) -> bool:
    """Compare two holdout evaluations using RMSE as the primary score."""
    if not incumbent_metrics:
        return True

    candidate_rmse = candidate_metrics.get("rmse")
    incumbent_rmse = incumbent_metrics.get("rmse")

    if candidate_rmse is None:
        return False
    if incumbent_rmse is None:
        return True

    return float(candidate_rmse) < float(incumbent_rmse)
