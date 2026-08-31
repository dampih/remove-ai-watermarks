"""Dola AI visible watermark detector/localizer.

Dola's image mark is a light gray ``Dola AI`` wordmark in the bottom-right.
The detector uses the shared one-sweep TextMarkEngine and a synthetic Arial Bold
silhouette; removal is the shared localize -> footprint mask -> fill pipeline.
"""
# pyright: reportUnusedFunction=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkEngine

if TYPE_CHECKING:
    from numpy.typing import NDArray

# The sample is 1024x575: the visible wordmark is about 103x22 px and sits
# 35 px from the right edge / 12 px from the bottom edge. Keep the locate box
# generous enough for rerasterization, while using short-side scaling.
WM_WIDTH_FRAC = 0.34
WM_HEIGHT_FRAC = 0.11
MARGIN_RIGHT_FRAC = 0.02
MARGIN_BOTTOM_FRAC = 0.012

# Dola uses a light, low-saturation gray overlay like other AIGC marks.
MAX_SATURATION = 55
LOGO_MIN_LUMA = 145
TOPHAT_DELTA = 10

# Conservative gate: the repository's Dola cohort has weak separation. Strict
# only avoids provenance turning unrelated bottom-right texture into a fill.
DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.52

# Asset is 146x32 (Arial Bold ``Dola AI``), scaled against short side.
_ALPHA_WIDTH_FRAC = 0.254
_ALPHA_HEIGHT_FRAC = 0.0557

_CONFIG = TextMarkConfig(
    name="Dola AI",
    asset_name="dola_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    detect_frontend="tophat",
    scale_basis="short",
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    min_gw=8,
    provenance_ncc_factor=1.0,
)


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Dola AI alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary Dola AI silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], scale_base: int) -> float:
    """TM_CCOEFF_NORMED of the Dola AI glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, scale_base, _CONFIG)


class DolaEngine(TextMarkEngine):
    """Detect/localize the visible Dola AI watermark."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)
