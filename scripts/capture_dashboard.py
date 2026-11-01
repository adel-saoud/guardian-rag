"""Re-capture `docs/dashboard.png` from the running dashboard.

Steps: ingest the 30 golden documents with the dashboard's detector, so
the stats panel has data, start Streamlit, open each tab in the
installed Google Chrome through Playwright, run the redaction explorer
once, and lay the panels out with a label above each: the explorer
across the top, the stats and eval panels below.

Usage:
    uv sync --extra ml --extra dashboard
    uv run --with playwright python scripts/capture_dashboard.py

Playwright drives the system Chrome (`channel="chrome"`), so no browser
download is needed. Run it after `scripts/reproduce.py`, so the picture
matches the current code.

Where a browser cannot be automated, take the three screenshots by hand:
    uv run python scripts/capture_dashboard.py --seed-only
    uv run streamlit run src/guardian_rag/dashboard/app.py
    uv run python scripts/capture_dashboard.py --compose stats.png explorer.png eval.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from PIL import Image, ImageDraw, ImageFont

from guardian_rag.config import Settings
from guardian_rag.dashboard.detector import build_dashboard_detector
from guardian_rag.eval.loader import load_golden_corpus
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.models import Document
from guardian_rag.ingestion.pipeline import IngestionPipeline, NullVectorSink
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.redactor import Redactor

if TYPE_CHECKING:
    from playwright.sync_api import Page

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "src" / "guardian_rag" / "dashboard" / "app.py"
REDACTION_DIR = ROOT / "data" / "redaction_maps"
OUT = ROOT / "docs" / "dashboard.png"
PORT = 8599
VIEWPORT = {"width": 1280, "height": 900}

PANELS = [
    ("PII Protection Stats", "PII protection stats"),
    ("Redaction Explorer", "Redaction explorer"),
    ("Eval Results", "Eval results"),
]
"""(tab name in the app, label drawn above the panel)."""

TILE_WIDTH = 1200
GAP = 8
LABEL_HEIGHT = 36
BACKGROUND_RGB = (255, 255, 255)
SEPARATOR_RGB = (220, 220, 220)
LABEL_TEXT_RGB = (40, 40, 40)
TOOLBAR_SCAN_WIDTH_RATIO = 0.8
"""Content-top search ignores the right 20%, where Streamlit's Deploy toolbar sits."""
MARGIN_RATIO = 0.02
"""White space kept around the content rows, as a share of the image height."""


def _seed_redaction_maps() -> None:
    """Replace the redaction maps with the golden corpus, as the dashboard pipeline writes them."""
    shutil.rmtree(REDACTION_DIR, ignore_errors=True)
    pipeline = IngestionPipeline(
        detector=build_dashboard_detector(Settings()),
        redactor=Redactor(),
        chunker=HierarchicalChunker(),
        redaction_store=RedactionMapStore(REDACTION_DIR),
        vector_sink=NullVectorSink(),
    )
    for doc in load_golden_corpus(ROOT / "eval" / "golden"):
        pipeline.ingest(Document(document_id=doc.document_id, text=doc.text, source="golden"))


def _wait_for_streamlit(timeout_s: float = 60.0) -> None:
    """Poll Streamlit's health endpoint until the server answers."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://localhost:{PORT}/_stcore/health", timeout=2.0).is_success:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not start on port {PORT} within {timeout_s:.0f}s")


def _capture(page: Page, tab: str, path: Path) -> None:
    """Open one tab, let it render, and screenshot the page."""
    page.get_by_role("tab", name=tab).click()
    if tab == "Redaction Explorer":
        page.get_by_role("button", name="Run redaction").click()
        page.get_by_text("Entities detected").wait_for()
    page.wait_for_load_state("networkidle")
    time.sleep(1.5)  # Plotly and dataframes paint after the network goes idle.
    page.screenshot(path=path, full_page=True)


def _content_rows(src: Image.Image) -> tuple[int, int]:
    """Rows from the first to the last line of content, plus a margin on both sides.

    A content row holds more than one colour, so the white page, a screenshot's
    border line and the Deploy toolbar (in the ignored right part) are all trimmed.
    """
    scan = src.crop((0, 0, int(src.width * TOOLBAR_SCAN_WIDTH_RATIO), src.height))
    rows = [
        y for y in range(scan.height) if scan.crop((0, y, scan.width, y + 1)).getcolors(1) is None
    ]
    if not rows:
        return 0, src.height
    margin = int(src.height * MARGIN_RATIO)
    return max(0, rows[0] - margin), min(src.height, rows[-1] + margin)


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, 22)
    return ImageFont.load_default()


def _tile(
    shot: Path, label: str, width: int, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> Image.Image:
    """One panel trimmed to its content, scaled to `width`, under a label strip."""
    src = Image.open(shot).convert("RGB")
    top, bottom = _content_rows(src)
    src = src.crop((0, top, src.width, bottom))
    panel = src.resize((width, int(src.height * width / src.width)), Image.LANCZOS)
    tile = Image.new("RGB", (width, LABEL_HEIGHT + panel.height), BACKGROUND_RGB)
    draw = ImageDraw.Draw(tile)
    draw.text((16, 6), label, fill=LABEL_TEXT_RGB, font=font)
    draw.line([(0, LABEL_HEIGHT - 1), (width, LABEL_HEIGHT - 1)], fill=SEPARATOR_RGB)
    tile.paste(panel, (0, LABEL_HEIGHT))
    return tile


def _compose(shots: list[Path], labels: list[str], out: Path) -> None:
    """The redaction explorer across the top, the stats and eval panels below it.

    Two columns keep each panel readable at GitHub's README width; one row of
    three made the text too small. The explorer gets the full width because it
    is the panel that shows what Guardian does.
    """
    font = _font()
    full_width = 2 * TILE_WIDTH + GAP
    stats, explorer, evals = (
        _tile(shot, label, full_width if i == 1 else TILE_WIDTH, font)
        for i, (shot, label) in enumerate(zip(shots, labels, strict=True))
    )
    bottom_height = max(stats.height, evals.height)
    canvas = Image.new(
        "RGB",
        (full_width + 2 * GAP, explorer.height + bottom_height + 3 * GAP),
        SEPARATOR_RGB,
    )
    canvas.paste(explorer, (GAP, GAP))
    bottom_y = explorer.height + 2 * GAP
    for x, tile in ((GAP, stats), (2 * GAP + TILE_WIDTH, evals)):
        padded = Image.new("RGB", (TILE_WIDTH, bottom_height), BACKGROUND_RGB)
        padded.paste(tile, (0, 0))
        canvas.paste(padded, (x, bottom_y))
    canvas.save(out, optimize=True)
    print(f"wrote {out.relative_to(ROOT)} ({canvas.width}x{canvas.height})")


def main() -> int:
    """Seed, serve, capture, compose; always stop the Streamlit server."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--seed-only",
        action="store_true",
        help="Only ingest the golden corpus, for a dashboard screenshotted by hand.",
    )
    mode.add_argument(
        "--compose",
        nargs=len(PANELS),
        type=Path,
        metavar="PNG",
        help="Only compose hand-taken screenshots, in tab order.",
    )
    args = parser.parse_args()
    if args.compose:
        _compose(args.compose, [label for _tab, label in PANELS], OUT)
        return 0
    _seed_redaction_maps()
    if args.seed_only:
        return 0
    from playwright.sync_api import sync_playwright

    server = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP),
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_streamlit()
        with tempfile.TemporaryDirectory() as tmp, sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome")
            page = browser.new_page(viewport=VIEWPORT)
            page.goto(f"http://localhost:{PORT}")
            page.get_by_role("tab", name=PANELS[0][0]).wait_for()
            shots = [Path(tmp) / f"panel{i}.png" for i in range(len(PANELS))]
            for (tab, _label), shot in zip(PANELS, shots, strict=True):
                _capture(page, tab, shot)
            browser.close()
            _compose(shots, [label for _tab, label in PANELS], OUT)
    finally:
        server.terminate()
        server.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
