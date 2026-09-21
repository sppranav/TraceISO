"""Plotly configuration for TraceISO."""

from typing import Optional

# Legacy/general Plotly sizes. Results figures override these through the named
# screen/export profiles in ``ui.theme`` so download typography is independent
# from the compact on-screen view.
PLOTLY_BASE_FONT_SIZE = 16
PLOTLY_AXIS_TITLE_FONT_SIZE = 18
PLOTLY_TITLE_FONT_SIZE = 20
PLOTLY_LEGEND_FONT_SIZE = 15
PLOTLY_ANNOTATION_FONT_SIZE = 15

WIDE_TIMESERIES_MAX_HEIGHT = 260
DRIFT_PREVIEW_HEIGHT = 620
DRIFT_PREVIEW_COMPACT_HEIGHT = 460
BALANCED_DISTRIBUTION_MIN_HEIGHT = 440


def get_plotly_config(
    filename: str = "traceiso_export",
    format: Optional[str] = None,
    *,
    width: int = 1400,
    height: int = 900,
) -> dict:
    """Return a standard Plotly configuration dict for clean toolbars and high-res export."""
    if format is None:
        try:
            from ui.utils import get_plot_config

            format = str(get_plot_config().get("download_format", "png"))
        except Exception:
            format = "png"
    if format not in {"png", "svg"}:
        raise ValueError("Plot export format must be 'png' or 'svg'.")
    image_options = {
        "format": format,
        "filename": filename,
        "height": height,
        "width": width,
        "setBackground": "white",
    }
    if format == "png":
        image_options["scale"] = 3  # Preserve print-quality resolution.
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
            "hoverCompareCartesian",
            "hoverClosestCartesian",
            "toggleSpikelines"
        ],
        "toImageButtonOptions": image_options,
    }
