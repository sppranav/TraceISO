"""Application defaults and configuration constants."""

APP_VERSION = "1.0.0"

COVERAGE_FACTOR_K = 2.0
DEFAULT_OUTLIER_THRESHOLD_SD = 2.0
DEFAULT_CONFIDENCE_LEVEL = 0.95

MIN_CYCLES_FOR_STATISTICS = 3

DEFAULT_DISPLAY_PRECISION = 6

DELTA_PER_MIL = 1000.0

PPM_TO_FRACTION = 1e-6
FRACTION_TO_PPM = 1e6

DEFAULT_EXCEL_COLUMN_WIDTH = 12
MAX_EXCEL_COLUMN_WIDTH = 30

# Literature reference material used for the Engine A (Sr) u_ref_value
# contributor — a static Type B term on the accepted GeoReM consensus ratio,
# independent of whichever CRM the user selected for anchoring (u_crm).
SR_GEOREM_REFERENCE_MATERIAL = "GeoReM NIST SRM 987"


# ---------------------------------------------------------------------------
# Sr refinement stability warning (A025)
# ---------------------------------------------------------------------------
# The Sr chain runs a natural-ratio initialization followed by exactly two
# K-factor refinements (domain/corrections/sr_chain.py). The relative movement of
# the final 87Sr/86Sr between the two refinements is reported for every sample,
# and a warning is raised when it exceeds this threshold.
#
# The value is the 0.1 ppm figure previously declared as the retired solver's
# final-ratio accuracy target (two orders of magnitude below a 10 ppm 87Sr/86Sr
# standard uncertainty), repurposed explicitly as an engineering reporting
# threshold. It is not an accuracy bound or a validated laboratory envelope: a
# movement below it does not show the result is accurate or on the physical root.
# It never rejects a result and never enters the uncertainty budget. A
# laboratory-approved threshold can replace it.
SR_REFINEMENT_STABILITY_WARNING_REL = 1.0e-7
