"""One-time script to generate crm_library.json from all available sources."""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import openpyxl
from tools.crm_manager.models.crm_data import (
    CRMLibrary, CertifiedRatio, InternalNormalization, NaturalRatio, ReferenceData,
    ReferenceMaterial,
    load_library,
    save_library,
)


_CERTIFIED_DATA_PATH = ROOT / "project_support" / "_archive" / "certified_data.json"
_AME2020_PATH = ROOT / "project_support" / "AME2020_Natural_Isotopes_Full_Precision.xlsx"
_CURRENT_LIBRARY_PATH = ROOT / "config" / "crm_library.json"

# Declared element-level reference ratios.
#
# These are reconciled with the shipped ``config/crm_library.json``: this file
# used to carry a second, older copy of the same quantities (87Rb/85Rb =
# 0.385670, 84Kr/83Kr = 4.955478, 86Kr/83Kr = 1.502783), so rerunning the
# generator would have silently reverted the library to superseded numbers.
# ``verify_declared_ratios_match_library`` below fails loudly if they drift
# apart again.
_REFERENCE_NATURAL_RATIOS = {
    "Rb": {
        # Nominal value derived from the CIAAW representative abundances
        # x(85Rb) = 0.7217(2), x(87Rb) = 0.2783(2). The uncertainty remains
        # unassigned for this release.
        "87Rb/85Rb": NaturalRatio(
            value=0.385617,
            uncertainty=None,
            k=None,
            source=(
                "CIAAW representative isotopic abundances x(85Rb) = 0.7217(2), "
                "x(87Rb) = 0.2783(2)"
            ),
            record_id="nat:rb:87rb-85rb",
            display_label="87Rb/85Rb representative isotopic abundance ratio",
            value_kind="representative_abundance",
            uncertainty_semantics="unassigned",
            source_url="https://www.ciaaw.org/isotopic-abundances.htm",
        ),
    },
    "Kr": {
        "84Kr/83Kr": NaturalRatio(
            value=4.955391,
            uncertainty=0.00827564826,
            k=1.0,
            source="Meija et al. (2016), IUPAC isotopic compositions of the elements",
            record_id="nat:kr:84kr-83kr",
            display_label="84Kr/83Kr representative isotopic abundance ratio",
            value_kind="representative_abundance",
            uncertainty_semantics="standard_uncertainty",
        ),
        "86Kr/83Kr": NaturalRatio(
            value=1.50252,
            uncertainty=0.00434304287,
            k=1.0,
            source="Meija et al. (2016), IUPAC isotopic compositions of the elements",
            record_id="nat:kr:86kr-83kr",
            display_label="86Kr/83Kr representative isotopic abundance ratio",
            value_kind="representative_abundance",
            uncertainty_semantics="standard_uncertainty",
        ),
    },
    "Sr": {
        # Stored with u = 0: treated as exact where used, not unknown.
        "84Sr/86Sr": NaturalRatio(
            value=0.05655,
            uncertainty=0.0,
            k=1.0,
            source="NIST SRM 987",
            record_id="nat:sr:84sr-86sr",
            display_label="84Sr/86Sr assumed reference ratio",
            value_kind="normalization_convention",
            uncertainty_semantics="assigned_exact",
        ),
        "88Sr/86Sr": NaturalRatio(
            value=8.37861,
            uncertainty=0.0,
            k=1.0,
            source="NIST SRM 987",
            record_id="nat:sr:88sr-86sr",
            display_label="88Sr/86Sr assumed reference ratio",
            value_kind="normalization_convention",
            uncertainty_semantics="assigned_exact",
        ),
    },
}


def verify_declared_ratios_match_library(library: CRMLibrary) -> None:
    """Fail if a declared ratio above disagrees with the shipped library.

    The generator rebuilds ``crm_library.json``, so a stale constant here is a
    silent scientific regression rather than a cosmetic one. Any disagreement
    has to be resolved by a person - with a source - before the rebuild runs.
    """
    mismatches = []
    for symbol, declared_ratios in _REFERENCE_NATURAL_RATIOS.items():
        element = library.get_element(symbol)
        stored = (
            element.reference_data.natural_ratios
            if element is not None and element.reference_data is not None
            else {}
        )
        for ratio_name, declared in declared_ratios.items():
            current = stored.get(ratio_name)
            if current is None:
                continue
            if (
                current.value != declared.value
                or current.uncertainty != declared.uncertainty
                or current.k != declared.k
            ):
                mismatches.append(
                    f"  {symbol} {ratio_name}: library has "
                    f"({current.value!r}, {current.uncertainty!r}, {current.k!r}), "
                    f"generator declares "
                    f"({declared.value!r}, {declared.uncertainty!r}, {declared.k!r})"
                )
    if mismatches:
        raise ValueError(
            "populate_library.py declares reference ratios that disagree with "
            "config/crm_library.json. Running the rebuild would overwrite the "
            "library with these values. Reconcile them against a source first:\n"
            + "\n".join(mismatches)
        )

_FIXED_REFERENCE_MASSES = {
    "Sr": {
        "84Sr": 83.9134191180,
        "86Sr": 85.9092607240,
        "87Sr": 86.9088774950,
        "88Sr": 87.9056122530,
    },
    "Rb": {
        "85Rb": 84.9117897380,
        "87Rb": 86.9091805310,
    },
    "Kr": {
        "83Kr": 82.914137,
        "84Kr": 83.911508,
        "86Kr": 85.9106106430,
    },
}


def _require_path(path: Path, description: str) -> Path:
    """Return *path* or raise a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} not found at '{path}'. "
            "Restore the source file or update populate_library.py path resolution."
        )
    return path


def load_ame2020_masses() -> dict:
    """Load atomic masses from AME2020 Excel."""
    xlsx = _require_path(_AME2020_PATH, "AME2020 isotope workbook")
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        ws = wb["All_Natural_Isotopes"]

        masses = {}  # {element: {isotope: mass}}
        for row in ws.iter_rows(min_row=2, values_only=True):
            element = row[1]   # Element symbol
            isotope = row[4]   # e.g. "84Sr"
            mass = row[6]      # Atomic_Mass_u
            if element and isotope and mass is not None:
                masses.setdefault(element, {})[isotope] = float(mass)
    finally:
        wb.close()
    for element, fixed_masses in _FIXED_REFERENCE_MASSES.items():
        masses.setdefault(element, {}).update(fixed_masses)
    return masses


def load_existing_library() -> CRMLibrary:
    """Load the current managed CRM library as the rebuild baseline."""
    return load_library(_CURRENT_LIBRARY_PATH)


def load_certified_data_json() -> dict:
    """Load certified_data.json."""
    path = _require_path(_CERTIFIED_DATA_PATH, "certified_data.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Mapping from certified_data.json keys to (element, crm_name, source)
_JSON_KEY_MAP = {
    "Li_LSVEC":           ("Li", "NIST RM 8545 (LSVEC)", "Fleischer (1973)"),
    "B_NIST_SRM_951a":    ("B",  "NIST SRM 951a",        "NIST Certificate"),
    "Mg_AE_143":          ("Mg", "ERM-AE143",             "ERM Certificate"),
    "Ni_NIST_SRM_986":    ("Ni", "NIST SRM 986",          "NIST Certificate"),
    "Ni_HINI_1":          ("Ni", "HINI-1",                "certified value"),
    "Ni_HICU_1":          ("Cu", "HICU-1",                "certified value"),
    "Sr_NIST_SRM_987":    ("Sr", "NIST SRM 987",          "NIST Certificate"),
    "Nd_NIST_SRM_3135a":  ("Nd", "NIST SRM 3135a",        "NIST Certificate"),
    "Hf_NIST_SRM_3122":   ("Hf", "NIST SRM 3122",         "NIST Certificate"),
    "Tl_NIST_SRM_997_IUPAC": (
        "Tl", "NIST SRM 997",
        "NIST (2004), Certificate of Analysis, SRM 997 Isotopic Standard for Thallium",
    ),
    "Cr_SRM_979":         ("Cr", "NIST SRM 979",           "NIST Certificate"),
    "Pb_SRM_981":         ("Pb", "NIST SRM 981",           "NIST Certificate"),
    "U_NBL_C145":         ("U",  "NBL CRM C145",           "NBL Certificate"),
    "Cd_BAM_I012":        ("Cd", "BAM-I012",               "BAM Certificate"),
}


def build_library() -> CRMLibrary:
    """Merge all sources into a single CRMLibrary."""
    lib = load_existing_library()
    verify_declared_ratios_match_library(lib)
    ame_masses = load_ame2020_masses()

    # Source 2: certified_data.json
    cert_data = load_certified_data_json()
    for json_key, data in cert_data.items():
        if json_key not in _JSON_KEY_MAP:
            print(f"  SKIP: {json_key} (no mapping)")
            continue

        element, crm_name, source = _JSON_KEY_MAP[json_key]

        # Get or create CRM
        existing = lib.get_crm(element, crm_name)
        if existing:
            # Merge ratios — add any that are missing
            for rname, (val, unc) in data.items():
                if rname not in existing.ratios:
                    existing.ratios[rname] = CertifiedRatio(
                        value=val, uncertainty=unc, k=2.0,
                    )
                    print(f"  ADD: {element}/{crm_name} -> {rname} = {val}")
        else:
            # Create new CRM
            ratios = {}
            for rname, (val, unc) in data.items():
                ratios[rname] = CertifiedRatio(
                    value=val, uncertainty=unc, k=2.0,
                )

            # Use full natural-isotope mass set for this element.
            filtered_masses = dict(ame_masses.get(element, {}))

            crm = ReferenceMaterial(
                name=crm_name,
                element=element,
                source=source,
                ratios=ratios,
                masses=filtered_masses,
            )
            lib.add_crm(element, crm)
            print(f"  NEW: {element}/{crm_name} ({len(ratios)} ratios)")

    # Seed conventional internal normalization for Sr.
    sr_entry = lib.get_element("Sr")
    if sr_entry is not None and sr_entry.internal_normalization is None:
        normalization_value = None
        sr_crm = sr_entry.reference_materials.get("NIST SRM 987")
        if sr_crm is not None:
            payload = sr_crm.ratios.get("88Sr/86Sr")
            if payload is not None and payload.value:
                normalization_value = 1.0 / float(payload.value)
        if normalization_value is None:
            normalization_value = 0.1194
        sr_entry.internal_normalization = InternalNormalization(
            ratio_name="86Sr/88Sr",
            value=normalization_value,
        )

    # Seed element-level reference_data.masses for all elements already in the
    # library, then add interference species with reference_data only.
    #
    # Natural ratios already stored are kept as they are. This block used to
    # replace the whole ``natural_ratios`` mapping with the declared table
    # above, which deleted every element not listed there - Hg, Tl and Zr
    # among them - and dropped the record ids, value kinds and uncertainty
    # semantics of the ones it did keep.
    for symbol in sorted(set(lib.element_symbols()) | {"Rb", "Kr"}):
        elem = lib.add_element(symbol)
        existing = (
            dict(elem.reference_data.natural_ratios)
            if elem.reference_data is not None
            else {}
        )
        for ratio_name, ratio in _REFERENCE_NATURAL_RATIOS.get(symbol, {}).items():
            existing.setdefault(ratio_name, ratio)
        elem.reference_data = ReferenceData(
            masses=dict(ame_masses.get(symbol, {})),
            natural_ratios=existing,
        )

    return lib


def main():
    raise RuntimeError(
        "The historical population utility is unsupported in the source toolkit: "
        "its reviewed maintenance input bundle is absent. Use CRM Manager with "
        "independently sourced records; this entry point will not rewrite the library."
    )
    print("Building CRM library...")
    lib = build_library()

    # Summary
    print(f"\nLibrary summary:")
    for sym in lib.element_symbols():
        elem = lib.get_element(sym)
        for crm_name, crm in elem.reference_materials.items():
            n_ratios = len(crm.ratios)
            n_masses = len(crm.masses)
            print(f"  {sym:3s} | {crm_name:30s} | {n_ratios} ratios, {n_masses} masses")

    out_path = ROOT / "config" / "crm_library.json"
    save_library(lib, out_path)
    print(f"\nSaved to: {out_path}")
    print(f"Total: {len(lib.elements)} elements, {lib.total_crms()} CRMs")


if __name__ == "__main__":
    main()
