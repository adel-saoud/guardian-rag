"""Run the PII recall + precision eval over the 30-doc golden corpus.

By default Stage 1 (Presidio) runs over the corpus, which keeps the eval
hermetic. Pass `--with-gliner` to load the real GLiNER backend (slow on
first run because it downloads the model). Faithfulness is not run from
this script: it needs Ollama and is wired into the dashboard instead.

Usage:
    uv run python scripts/eval.py
    uv run python scripts/eval.py --docs eval/golden --report eval/report.html
    uv run python scripts/eval.py --with-gliner
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from guardian_rag.eval.loader import load_golden_corpus
from guardian_rag.eval.models import EvalReport
from guardian_rag.eval.runner import run_pii_eval
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.presidio_engine import PresidioEngine

console = Console()


def _build_detector(with_gliner: bool) -> PiiDetector:
    presidio = PresidioEngine()
    if not with_gliner:
        return PiiDetector(presidio=presidio)
    from guardian_rag.pii._real_backends import RealGlinerBackend

    backend = RealGlinerBackend()
    gliner = GlinerRecognizer(backend=backend)
    return PiiDetector(presidio=presidio, gliner=gliner)


def _format_table(report: EvalReport) -> Table:
    table = Table(title="Guardian RAG · PII detection eval")
    table.add_column("Slice")
    table.add_column("Planted", justify="right")
    table.add_column("Detected", justify="right")
    table.add_column("Recall hits", justify="right")
    table.add_column("Precision hits", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    for category, counts in sorted(report.per_category.items()):
        table.add_row(
            category,
            str(counts.planted),
            str(counts.detected),
            str(counts.recall_hits),
            str(counts.precision_hits),
            f"{counts.recall:.1%}",
            f"{counts.precision:.2f}",
        )
    table.add_section()
    table.add_row(
        "overall",
        str(report.overall.planted),
        str(report.overall.detected),
        str(report.overall.recall_hits),
        str(report.overall.precision_hits),
        f"{report.overall.recall:.1%}",
        f"{report.overall.precision:.2f}",
        style="bold",
    )
    return table


def _write_html_report(report: EvalReport, path: Path) -> None:
    rows = "".join(
        f"<tr><td>{cat}</td><td>{c.planted}</td><td>{c.detected}</td>"
        f"<td>{c.recall_hits}</td><td>{c.precision_hits}</td>"
        f"<td>{c.recall:.1%}</td><td>{c.precision:.2f}</td></tr>"
        for cat, c in sorted(report.per_category.items())
    )
    overall = report.overall
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Guardian RAG eval</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif;
        max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ margin-bottom: 0.5rem; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid #e2e8f0; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
tr.total {{ font-weight: 600; background: #f7fafc; }}
</style>
</head>
<body>
<h1>Guardian RAG, PII detection eval</h1>
<table>
<thead><tr><th>Slice</th><th>Planted</th><th>Detected</th>
<th>Recall hits</th><th>Precision hits</th>
<th>Recall</th><th>Precision</th></tr></thead>
<tbody>
{rows}
<tr class="total"><td>overall</td><td>{overall.planted}</td><td>{overall.detected}</td>
<td>{overall.recall_hits}</td><td>{overall.precision_hits}</td>
<td>{overall.recall:.1%}</td><td>{overall.precision:.2f}</td></tr>
</tbody></table>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("eval/golden"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--with-gliner",
        action="store_true",
        help="Load the real GLiNER backend (slower, more accurate).",
    )
    parser.add_argument(
        "--recall-gate",
        type=float,
        default=0.85,
        help="Minimum overall recall. Below this, exit non-zero.",
    )
    parser.add_argument(
        "--precision-gate",
        type=float,
        default=0.85,
        help="Minimum overall precision. Below this, exit non-zero.",
    )
    args = parser.parse_args()

    corpus = load_golden_corpus(args.docs)
    detector = _build_detector(with_gliner=args.with_gliner)
    report = run_pii_eval(corpus, detector)

    console.print(_format_table(report))
    console.print(
        f"\nPII Recall      {report.overall.recall:.1%}\n"
        f"PII Precision   {report.overall.precision:.2f}\n"
        f"Faithfulness    N/A  (informational, requires Ollama)\n"
        f"Leaks caught    {report.leaks_caught}  (post-generation audit)\n"
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_html_report(report, args.report)
        console.print(f"[dim]wrote {args.report}[/dim]")

    gates_ok = (
        report.overall.recall >= args.recall_gate
        and report.overall.precision >= args.precision_gate
    )
    if not gates_ok:
        console.print(
            f"[red]gate failed[/red] "
            f"(recall gate {args.recall_gate:.0%}, precision gate {args.precision_gate:.2f})"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
