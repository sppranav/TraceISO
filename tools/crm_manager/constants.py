"""Shared paths for the CRM manager."""

from pathlib import Path


DEFAULT_CRM_LIBRARY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "crm_library.json"
)
