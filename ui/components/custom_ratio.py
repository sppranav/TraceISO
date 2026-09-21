"""
Custom ratio creation for TraceISO.

Allows users to manually create ratios from available isotopes.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import streamlit as st

from domain.models import Sample, CycleData
from ui.state import get_state


def _coerce_selected_ratio_set(value) -> Set[str]:
    """Normalize arbitrary stored ratio selections into a set of strings."""
    if value is None:
        return set()
    if isinstance(value, set):
        return set(value)
    if isinstance(value, str):
        return {value}
    try:
        return set(value)
    except TypeError:
        return {str(value)}


def get_selected_ratios(samples: List[Sample]) -> Set[str]:
    """Get the set of ratios selected by the user.

    Returns all available ratios if never initialized (first load).
    Uses post-processing samples when available so ratio names match
    what the inspector/results tabs actually display.
    """
    # Use post-processing samples if available (ratio names may differ)
    state = get_state()
    has_result = bool(getattr(state, "has_result", False))
    result = getattr(state, "result", None)
    effective_samples = result.samples if has_result and result is not None else samples

    has_selection = bool(
        getattr(
            state,
            "has_selected_ratio_selection",
            st.session_state.get("selected_ratios") is not None,
        )
    )
    if has_selection:
        # User has interacted with the multiselect - use their selection
        # (even if empty - respect the user's choice!)
        return _coerce_selected_ratio_set(
            getattr(state, "selected_ratios", st.session_state.get("selected_ratios"))
        )

    # First load: return all available ratios as default
    all_ratios = set()
    for sample in effective_samples:
        if sample.ratios:
            all_ratios.update(sample.ratios.keys())
    return all_ratios


def render_custom_ratio_creator(
    samples: List[Sample],
    key: str = "custom_ratio",
) -> Optional[str]:
    """Render UI to create a custom ratio.

    Parameters
    ----------
    samples:
        List of samples to add custom ratio to.
    key:
        Unique key for the widget.

    Returns
    -------
    Name of created ratio if successful, None otherwise.
    """
    if not samples:
        return None
    state = get_state()
    target_samples = _get_ratio_target_samples(samples, state)

    available_isotopes = set()
    for sample in target_samples:
        available_isotopes.update(_available_isotopes_for_sample(sample))

    if not available_isotopes:
        st.warning("No isotopes available to create ratios.")
        return None

    element_config = getattr(state, "element_config", None)
    examples = _ratio_examples(available_isotopes, getattr(element_config, "symbol", None))
    example_text = ", ".join(examples) if examples else "Numerator/Denominator"

    # Creation is occasional; routine ratio selection stays visible above it.
    with st.expander("Create custom ratio", expanded=False):
        st.caption(f"Type ratio names separated by commas (e.g., {example_text}).")

        col1, col2 = st.columns([3, 1])

        with col1:
            ratio_input = st.text_input(
                "Ratio Names",
                placeholder=f"e.g., {example_text}",
                key=f"{key}_input",
                label_visibility="collapsed",
            )

        with col2:
            create_button = st.button(
                "➕ Add",
                type="primary",
                width="stretch",
                key=f"{key}_create",
            )

        if create_button:
            if not ratio_input.strip():
                st.warning("Enter at least one ratio before clicking Add.")
                return None

            # Parse comma-separated ratios
            ratio_inputs = [r.strip() for r in ratio_input.split(",") if r.strip()]

            if not ratio_inputs:
                st.error("Please enter at least one ratio")
                return None

            created_ratios = []
            for single_ratio in ratio_inputs:
                result = _create_single_ratio(target_samples, single_ratio, available_isotopes)
                if result:
                    created_ratios.append(result)

            if created_ratios:
                # Auto-select newly created ratios and signal the ratio manager
                # to refresh its multiselect widget on the next rerun.
                selected = state.selected_ratios or set()
                state.selected_ratios = selected | set(created_ratios)
                st.session_state["_ratio_selection_external_update"] = True

                st.success(f"✓ Created {len(created_ratios)} ratio(s): {', '.join(created_ratios)}")
                return created_ratios[0]  # Return first one to trigger rerun
            return None

    return None


def _ratio_examples(
    available_isotopes: Set[str],
    element_symbol: Optional[str],
    limit: int = 2,
) -> List[str]:
    """Build input-format examples from the loaded isotopes, not a fixed element.

    Channels of the session's element are preferred over monitor channels; each
    example places a heavier isotope over the lightest one. Returns an empty
    list when fewer than two mass-labelled channels are available.
    """
    parsed = []
    for isotope in available_isotopes:
        match = re.fullmatch(r"(\d+)([A-Za-z]+)", str(isotope).strip())
        if match:
            parsed.append((int(match.group(1)), match.group(2), isotope))
    if element_symbol:
        own = [entry for entry in parsed if entry[1] == element_symbol]
        parsed = own or parsed
    if len(parsed) < 2:
        return []
    parsed.sort()
    reference = parsed[0][2]
    return [f"{isotope}/{reference}" for _, _, isotope in reversed(parsed[1:])][:limit]


def _get_ratio_target_samples(samples: List[Sample], state) -> List[Sample]:
    """Return raw and processed sample objects that should receive custom ratios."""
    # SANCTIONED MUTATION SITE: custom-ratio creation adds fresh CycleData keys
    # to both loaded samples and state.result.samples so every UI layer sees the
    # same user-defined ratio. Inspector edits are the only other direct sample
    # mutation path; keep this target list centralized here.
    target_samples: List[Sample] = list(samples)
    result = getattr(state, "result", None)
    if getattr(state, "has_result", False) and result is not None:
        target_samples.extend(getattr(result, "samples", []) or [])

    seen: Set[int] = set()
    unique: List[Sample] = []
    for sample in target_samples:
        identity = id(sample)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(sample)
    return unique


def _available_isotopes_for_sample(sample: Sample) -> Set[str]:
    """Return isotope channels available on any intensity layer for *sample*."""
    isotopes: Set[str] = set()
    for attr in ("intensities", "blank_corrected_intensities", "corrected_intensities"):
        layer = getattr(sample, attr, None)
        if layer:
            isotopes.update(layer.keys())
    return isotopes


def _compute_ratio_cycle_data(
    intensities: Dict[str, CycleData],
    numerator: str,
    denominator: str,
) -> Tuple[Optional[CycleData], int, int]:
    """Compute a ratio from an intensity layer and report masked-cycle counts."""
    if numerator not in intensities or denominator not in intensities:
        return None, 0, 0

    num_data = intensities[numerator]
    den_data = intensities[denominator]
    zero_denom = np.isclose(den_data.values, 0.0)
    invalid_inputs = ~np.isfinite(num_data.values) | ~np.isfinite(den_data.values)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = num_data.values / den_data.values

    ratio_values = np.asarray(ratio_values, dtype=float)
    non_finite_ratio = ~np.isfinite(ratio_values)
    invalid_cycles = zero_denom | invalid_inputs | non_finite_ratio
    ratio_values[invalid_cycles] = np.nan

    ratio_mask = num_data.mask & den_data.mask & ~invalid_cycles
    return (
        CycleData(values=ratio_values, mask=ratio_mask),
        int(np.sum(zero_denom)),
        int(np.sum(invalid_cycles & ~zero_denom)),
    )


def _add_ratio_to_sample_layers(
    sample: Sample,
    numerator: str,
    denominator: str,
    ratio_name: str,
) -> Tuple[int, int, int]:
    """Add a ratio to raw, blank-corrected, and corrected layers when possible."""
    added = 0
    zero_denom_count = 0
    non_finite_count = 0
    layer_pairs = (
        ("intensities", "ratios"),
        ("blank_corrected_intensities", "blank_corrected_ratios"),
        ("corrected_intensities", "corrected_ratios"),
    )

    for source_attr, target_attr in layer_pairs:
        source = getattr(sample, source_attr, None)
        target = getattr(sample, target_attr, None)
        if not source or target is None or ratio_name in target:
            continue

        ratio_cd, zero_count, non_finite = _compute_ratio_cycle_data(
            source,
            numerator,
            denominator,
        )
        if ratio_cd is None:
            continue

        target[ratio_name] = ratio_cd
        added += 1
        zero_denom_count += zero_count
        non_finite_count += non_finite

    return added, zero_denom_count, non_finite_count


def _create_single_ratio(
    samples: List[Sample],
    ratio_input: str,
    available_isotopes: Set[str],
) -> Optional[str]:
    """Create a single ratio from input string.

    Returns the ratio name if successful, None otherwise.
    """
    if not ratio_input:
        return None

    # Parse ratio name
    if "/" not in ratio_input:
        st.error(f"'{ratio_input}' - Ratio must be in format: Numerator/Denominator (e.g., 208Pb/206Pb)")
        return None

    parts = ratio_input.split("/")
    if len(parts) != 2:
        st.error(f"'{ratio_input}' - Ratio must have exactly one '/' separator")
        return None

    numerator = parts[0].strip()
    denominator = parts[1].strip()

    if numerator == denominator:
        st.error(f"'{ratio_input}' - Numerator and denominator must be different!")
        return None

    if numerator not in available_isotopes:
        st.error(f"'{ratio_input}' - Isotope '{numerator}' not found. Available: {', '.join(sorted(available_isotopes))}")
        return None

    if denominator not in available_isotopes:
        st.error(f"'{ratio_input}' - Isotope '{denominator}' not found. Available: {', '.join(sorted(available_isotopes))}")
        return None

    ratio_name = f"{numerator}/{denominator}"

    existing_ratios = {
        ratio_key
        for sample in samples
        for layer in (
            sample.ratios,
            sample.blank_corrected_ratios,
            sample.corrected_ratios,
        )
        for ratio_key in layer.keys()
    }

    # Calculate ratio for all samples/layers.
    n_added = 0
    n_masked_zero_denom = 0
    n_masked_non_finite = 0
    for sample in samples:
        added, zero_count, non_finite = _add_ratio_to_sample_layers(
            sample,
            numerator,
            denominator,
            ratio_name,
        )
        n_added += added
        n_masked_zero_denom += zero_count
        n_masked_non_finite += non_finite

    if n_added > 0:
        if n_masked_zero_denom > 0:
            st.warning(
                f"Masked {n_masked_zero_denom} cycle(s) with zero denominator "
                f"while creating '{ratio_name}'."
            )
        if n_masked_non_finite > 0:
            st.warning(
                f"Masked {n_masked_non_finite} additional non-finite cycle(s) "
                f"while creating '{ratio_name}'."
            )
        return ratio_name
    if ratio_name in existing_ratios:
        # Check if it's just filtered out (not selected in multiselect)
        state = get_state()
        selected = state.selected_ratios
        if ratio_name not in selected:
            # Auto-select it and signal the ratio manager to refresh its widget.
            state.selected_ratios = selected | {ratio_name}
            st.session_state["_ratio_selection_external_update"] = True
            st.success(f"'{ratio_name}' already exists - now added to selected ratios")
            return ratio_name  # Return to trigger rerun

        st.info(f"'{ratio_name}' already exists and is already selected")
        return None

    st.error(f"'{ratio_name}' - Could not create ratio, missing isotope data")
    return None


def render_ratio_manager(
    samples: List[Sample],
    key: str = "ratio_manager",
) -> None:
    """Show available ratios and allow selection."""
    if not samples:
        return

    # Use post-processing samples if available (ratio names may differ)
    state = get_state()
    effective_samples = state.result.samples if state.has_result else samples


    all_ratios = set()
    for sample in effective_samples:
        if sample.ratios:
            all_ratios.update(sample.ratios.keys())
    # Also include ratios from pre-processing samples (custom ratios may
    # only exist there before re-processing)
    for sample in samples:
        if sample.ratios:
            all_ratios.update(sample.ratios.keys())

    if not all_ratios:
        st.info("No ratios available. Load data first.")
        return

    ratio_list = sorted(all_ratios)

    if not state.has_selected_ratio_selection:
        # First load only: select all ratios as default
        state.selected_ratios = set(ratio_list)
    else:
        state.selected_ratios = _coerce_selected_ratio_set(state.selected_ratios)

    # Only force the multiselect widget to re-render when external code
    # (e.g. the custom ratio creator) has explicitly signalled an update.
    # Using session_val - widget_val was buggy: that condition is also true
    # when the user *deselects* a ratio (session_val is still the old larger
    # set, widget_val is the new smaller set), which caused the deselection
    # to be silently undone on every rerun.
    multiselect_key = f"{key}_multiselect"
    _external_update = st.session_state.pop("_ratio_selection_external_update", False)
    if _external_update and multiselect_key in st.session_state:
        del st.session_state[multiselect_key]

    # Multiselect for ratio selection
    st.caption(
        "**Ratios available for review and export** — limits ratio choices in "
        "Sample Inspector, Instrumental Drift, Results & Statistics, "
        "Uncertainty Budgets, and Export."
    )
    selected = st.multiselect(
        "Active Ratios",
        options=ratio_list,
        default=list(state.selected_ratios & set(ratio_list)),
        key=multiselect_key,
        label_visibility="collapsed",
    )

    state.selected_ratios = set(selected)

    st.caption(f"Total: {len(ratio_list)} available | {len(selected)} selected")

