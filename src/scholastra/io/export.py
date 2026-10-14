"""Pandoc driver: canonical Markdown -> PDF / DOCX, with graceful degradation."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolStatus:
    pandoc: str | None = None
    pdf_engine: str | None = None

    @property
    def can_export_docx(self) -> bool:
        return bool(self.pandoc)

    @property
    def can_export_pdf(self) -> bool:
        return bool(self.pandoc and self.pdf_engine)

    def describe(self) -> str:
        pandoc = self.pandoc or "not found"
        engine = self.pdf_engine or "not found"
        return f"pandoc={pandoc}; pdf-engine={engine}"


@dataclass
class ExportResult:
    outputs: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.outputs)

    def summary(self) -> str:
        if not self.outputs:
            return "no exports produced"
        return ", ".join(sorted(self.outputs))


def detect_tools() -> ToolStatus:
    status = ToolStatus()
    status.pandoc = shutil.which("pandoc")
    for engine in ("xelatex", "pdflatex", "lualatex", "tectonic", "typst"):
        found = shutil.which(engine)
        if found:
            status.pdf_engine = found
            break
    return status


def build_pandoc_args(
    md_path: Path,
    output: Path,
    *,
    fmt: str,
    csl_path: Path | None = None,
    bib_path: Path | None = None,
    pdf_engine: str | None = None,
    resource_path: Path | None = None,
    toc: bool = True,
) -> list[str]:
    args = ["pandoc", str(md_path), "-o", str(output), "--from", "markdown+smart"]
    if toc and fmt in {"pdf", "docx", "html"}:
        args.append("--toc")
        args.extend(["--toc-depth", "3"])
    if bib_path is not None and bib_path.exists():
        args.extend(["--citeproc", "--bibliography", str(bib_path)])
        if csl_path is not None and csl_path.exists():
            args.extend(["--csl", str(csl_path)])
    if resource_path is not None:
        args.extend(["--resource-path", str(resource_path)])
    if fmt == "pdf":
        if pdf_engine:
            args.extend(["--pdf-engine", Path(pdf_engine).stem or Path(pdf_engine).name])
        args.extend(["-V", "geometry:margin=1in"])
    return args


def export_document(
    md_path: Path,
    out_dir: Path,
    *,
    formats: list[str],
    csl_path: Path | None = None,
    bib_path: Path | None = None,
    stem: str = "final",
    timeout: int = 300,
) -> ExportResult:
    """Render ``md_path`` into each requested format. Markdown is always kept."""
    md_path = Path(md_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(outputs={"md": str(md_path)})
    status = detect_tools()

    requested = [f for f in formats if f in {"pdf", "docx", "html"}]
    if requested and not status.pandoc:
        result.warnings.append(
            "Pandoc was not found on PATH; wrote Markdown only. "
            "Install Pandoc (and a LaTeX engine for PDF) to enable PDF/DOCX export."
        )
        return result

    for fmt in requested:
        if fmt == "pdf" and not status.pdf_engine:
            result.warnings.append(
                "No LaTeX PDF engine (xelatex/pdflatex) found; skipped PDF. "
                "Markdown and other exports are unaffected."
            )
            continue
        output = out_dir / f"{stem}.{fmt}"
        args = build_pandoc_args(
            md_path,
            output,
            fmt=fmt,
            csl_path=csl_path,
            bib_path=bib_path,
            pdf_engine=status.pdf_engine,
            resource_path=md_path.parent,
        )
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout, check=False
            )
        except Exception as exc:  # noqa: BLE001 - report, keep Markdown
            result.warnings.append(f"{fmt} export failed: {exc}")
            continue
        if proc.returncode == 0 and output.exists():
            result.outputs[fmt] = str(output)
        else:
            result.warnings.append(
                f"{fmt} export failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}"
            )
    return result
