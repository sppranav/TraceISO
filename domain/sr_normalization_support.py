"""Which Sr internal-normalization pairs TraceISO actually models (A039/D1).

The Sr correction chain is role-driven: ``ElementConfig.correction_roles`` names
which channel is the mass-bias normalization numerator and denominator, which is
the interference-corrected target, and which are the Rb/Kr monitors.
``ProcessingPipeline._sr_correction_loop`` corrects the role-designated channels
and forms the normalization ratio from the role-designated pair.

The analytical Kragten replay and the Monte Carlo chain replay do not follow the
roles that far.  ``domain.uncertainty.kragten._series_for_label`` resolves the
normalization channels by literal isotope label and deliberately returns an
*uncorrected* ``84Sr`` series, because the 84Kr correction does not feed back
into ``87Sr/86Sr``.  So when a session overrides the normalization pair away from
the declared roles, the three paths stop describing the same measurement:
production corrects one pair of channels, Kragten replays another, and the Monte
Carlo replay inherits Kragten's reference inputs.  That was A039 — not a
differently normalized result, but three mutually inconsistent models of one
session, none of them flagged.

D1 resolves it by restricting the supported envelope to the pair the active
element configuration is actually derived for, and refusing everything else
explicitly and identically in production, Kragten and Monte Carlo.  This is a
deliberate V1 scope reduction.  A refused selection is reported with a reason and
never coerced to the default, because silently substituting the supported pair
would report numbers the user did not ask for under the label they chose.

The envelope is expressed against the declared roles rather than a hard-coded
``86Sr/88Sr`` literal, so a role-consistent configuration with non-standard
isotope labels stays supported while a session override away from those roles
does not.  Every caller — production, Kragten, Monte Carlo, the eligibility
helpers and the session-configuration UI — must consult
:func:`sr_normalization_support` so the refusal cannot drift apart again.
"""

from __future__ import annotations

from typing import Optional, Tuple

from domain.ratio_utils import normalize_ratio_name, normalize_ratio_token

#: The Sr internal-normalization pair the shipped Sr configuration is derived for.
SUPPORTED_SR_NORMALIZATION_RATIO = "86Sr/88Sr"


def supported_sr_normalization_ratio(element_config: object) -> Optional[str]:
    """Return the one normalization pair ``element_config`` is derived for.

    Taken from the declared correction roles, falling back to the element's own
    ``normalization_ratio`` when no roles are declared.
    """
    roles = getattr(element_config, "correction_roles", None) or {}
    numerator = roles.get("normalization_numerator")
    denominator = roles.get("normalization_denominator")
    if numerator and denominator:
        return "{0}/{1}".format(
            normalize_ratio_token(str(numerator)),
            normalize_ratio_token(str(denominator)),
        )
    declared = getattr(element_config, "normalization_ratio", None)
    return normalize_ratio_name(declared) if declared else None


def is_supported_sr_normalization_ratio(
    element_config: object,
    ratio_name: Optional[str],
) -> bool:
    """Whether ``ratio_name`` is the pair ``element_config`` is derived for."""
    supported = supported_sr_normalization_ratio(element_config)
    if supported is None or not ratio_name:
        return True
    return normalize_ratio_name(ratio_name) == supported


def sr_normalization_support(
    element_config: object,
    ratio_name: Optional[str],
) -> Tuple[bool, str]:
    """Return ``(supported, reason)`` for one session's normalization selection.

    ``reason`` is empty when the selection is supported, and otherwise carries
    user-facing text explaining the refusal.  Elements other than Sr are not
    governed by this envelope and are always reported as supported.
    """
    if (getattr(element_config, "symbol", "") or "").strip() != "Sr":
        return True, ""
    if ratio_name is None or not str(ratio_name).strip():
        # No selection at all is a different condition, handled by the existing
        # missing-normalization-input path rather than treated as unsupported.
        return True, ""

    supported_ratio = supported_sr_normalization_ratio(element_config)
    if supported_ratio is None:
        return True, ""

    selected = normalize_ratio_name(ratio_name) or str(ratio_name)
    if selected == supported_ratio:
        return True, ""

    return False, (
        f"Sr internal normalization is supported only for {supported_ratio}, the "
        f"pair this session's correction roles are defined for. The selected pair "
        f"{selected} is refused because the Rb/Kr interference corrections and "
        f"the K-factor recursion correct the {supported_ratio} channels, while "
        f"the Kragten sensitivities and the Monte Carlo replay resolve the "
        f"normalization channels by isotope label. Accepting {selected} would "
        f"model the same session three different ways. Select {supported_ratio}, "
        f"or process this session without internal normalization."
    )


def require_supported_sr_normalization_ratio(
    element_config: object,
    ratio_name: Optional[str],
) -> None:
    """Raise :class:`UnsupportedSrNormalizationError` for a refused pair."""
    supported, reason = sr_normalization_support(element_config, ratio_name)
    if not supported:
        raise UnsupportedSrNormalizationError(reason)


class UnsupportedSrNormalizationError(ValueError):
    """An Sr normalization pair outside the supported envelope was requested.

    Raised rather than silently substituting the supported pair, so the refusal
    is visible wherever the session is processed, replayed or exported.
    """
