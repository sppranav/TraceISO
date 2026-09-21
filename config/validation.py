"""Shared JSON loading and validation helpers for configuration loaders."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import CustomUncertaintyContributor

_VALID_DISTRIBUTIONS = {"normal", "rectangular"}

VALID_CUSTOM_CONTRIBUTOR_NAME_RE = re.compile(r"^u_custom_[a-z0-9_]+$")


class DuplicateJSONMemberError(ValueError):
    """A JSON object repeated a member name, so a saved value would be lost."""


def _object_pairs_without_duplicates(pairs):
    """Build a dict from decoded member pairs, rejecting repeated names.

    ``json.loads`` keeps the last value for a repeated member, which silently
    discards the earlier one *before* any loader's duplicate check can see it.
    A copied or hand-merged lab file can therefore lose whole definitions. This
    hook runs on every object, so nested objects are covered too.
    """
    seen: set = set()
    for key, _value in pairs:
        if key in seen:
            raise DuplicateJSONMemberError(
                f"Duplicate JSON member name {key!r}; the file would silently "
                "lose one of the values."
            )
        seen.add(key)
    return dict(pairs)


def load_json_file(path: Path, name_label: str) -> dict:
    """Load a JSON file or raise ValueError with the file path and context."""
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs_without_duplicates,
        )
    except DuplicateJSONMemberError as exc:
        raise ValueError(f"Invalid {name_label} JSON at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid {name_label} JSON at {path}: {exc.msg}."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Could not read {name_label} JSON at {path}: {exc}."
        ) from exc


def normalize_distribution(value: object, *, field_name: str) -> str:
    """Normalise and validate a distribution string, mapping gaussian to normal."""
    distribution = str(value or "normal").strip().lower()
    if distribution == "gaussian":
        distribution = "normal"
    if distribution not in _VALID_DISTRIBUTIONS:
        raise ValueError(
            f"{field_name} distribution must be one of {_VALID_DISTRIBUTIONS}, "
            f"got {value!r}."
        )
    return distribution


def parse_nonnegative_finite_float(value: object, *, field_name: str) -> float:
    """Parse and validate a non-negative finite float."""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"{field_name} must be a non-negative finite value, got {value!r}."
        )
    return parsed


def parse_dof(value: object) -> float:
    """Parse and validate a degrees-of-freedom value (positive float or infinity)."""
    if isinstance(value, str) and value.strip().lower() in {"inf", "infinity"}:
        return float("inf")
    try:
        dof = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"degrees_of_freedom must be a number or 'inf', got {value!r}.") from exc
    if not math.isfinite(dof) or dof < 1:
        raise ValueError(
            f"degrees_of_freedom must be finite >= 1 or 'inf', got {value!r}."
        )
    return dof


def parse_custom_contributor_entries(
    element_symbol: str,
    entries: object,
) -> List["CustomUncertaintyContributor"]:
    """Parse one element's custom uncertainty contributors from JSON entries.

    Shared by the global-values loader and the standalone custom-contributor
    loader so a record cannot pass one and fail the other.

    ``u_rel_permil`` is required. An absent magnitude is not the same thing as
    zero: the lab's value is simply unknown, and substituting ``0.0`` would
    quietly drop a selected contributor out of the reported budget. An
    explicitly written ``0`` is still accepted, and stays distinguishable from
    absence because it had to be written down.
    """
    from config.contributor_names import parse_config_bool
    from config.settings import CustomUncertaintyContributor

    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(
            f"Custom contributors for {element_symbol!r} must be a JSON array."
        )

    seen: set[str] = set()
    parsed: List[CustomUncertaintyContributor] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Each custom contributor for {element_symbol!r} must be a JSON object."
            )
        name = str(entry.get("name") or "").strip()
        if not VALID_CUSTOM_CONTRIBUTOR_NAME_RE.match(name):
            raise ValueError(
                f"Custom contributor name {name!r} must match ^u_custom_[a-z0-9_]+$."
            )
        if name in seen:
            raise ValueError(
                f"Duplicate custom contributor name {name!r} for element {element_symbol!r}."
            )
        seen.add(name)

        if "u_rel_permil" not in entry:
            raise ValueError(
                f"Custom contributor {name!r} for element {element_symbol!r} requires "
                "u_rel_permil; an absent magnitude is not zero."
            )
        u_rel_permil = parse_nonnegative_finite_float(
            entry["u_rel_permil"],
            field_name=f"Custom contributor {name!r} u_rel_permil",
        )

        type_ab = str(entry.get("type_ab") or "B").strip().upper()
        if type_ab not in {"A", "B"}:
            raise ValueError(
                f"Custom contributor {name!r} type_ab must be 'A' or 'B'."
            )

        dof = parse_dof(entry.get("degrees_of_freedom", "inf"))
        if type_ab == "A" and not math.isfinite(dof):
            raise ValueError(
                f"Type A custom contributor {name!r} requires finite degrees_of_freedom."
            )

        distribution = normalize_distribution(
            entry.get("distribution", "normal"),
            field_name=f"Custom contributor {name!r}",
        )

        parsed.append(
            CustomUncertaintyContributor(
                name=name,
                display_name=str(entry.get("display_name") or name).strip() or name,
                element_symbol=element_symbol,
                u_rel_permil=u_rel_permil,
                type_ab=type_ab,
                degrees_of_freedom=dof,
                distribution=distribution,
                description=str(entry.get("description") or ""),
                reference=str(entry.get("source") or entry.get("reference") or "").strip(),
                enabled=parse_config_bool(
                    entry.get("enabled", True),
                    field_name=f"Custom contributor {name!r} enabled",
                ),
            )
        )
    return parsed
