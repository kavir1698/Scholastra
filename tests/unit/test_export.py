"""Export driver: Pandoc argument building, degradation, and real DOCX render."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from theo_explore.io import export as export_mod
from theo_explore.io.export import ToolStatus, build_pandoc_args, detect_tools, export_document

GOLDEN_MD = """---
title: Golden Document
---

# Golden Document

A short paragraph with a citation [@doi:1] and some text.

## References {#refs}

::: {#refs}
:::
"""


def test_build_pandoc_args_includes_toc_and_csl(tmp_path: Path):
    csl = tmp_path / "style.csl"
    bib = tmp_path / "refs.bib"
    csl.write_text("<style/>", encoding="utf-8")
    bib.write_text("", encoding="utf-8")
    args = build_pandoc_args(
        tmp_path / "in.md",
        tmp_path / "out.pdf",
        fmt="pdf",
        csl_path=csl,
        bib_path=bib,
        pdf_engine="C:/tex/xelatex.exe",
    )
    assert "--toc" in args
    assert "--citeproc" in args
    assert "--pdf-engine" in args
    assert args[args.index("--pdf-engine") + 1] == "xelatex"


def test_export_degrades_without_pandoc(tmp_path: Path, monkeypatch):
    md = tmp_path / "in.md"
    md.write_text(GOLDEN_MD, encoding="utf-8")
    monkeypatch.setattr(export_mod, "detect_tools", lambda: ToolStatus())
    result = export_document(md, tmp_path, formats=["pdf", "docx"])
    assert result.outputs == {"md": str(md)}
    assert result.warnings
    assert "Pandoc" in result.warnings[0]


def test_detect_tools_on_this_machine():
    status = detect_tools()
    assert isinstance(status, ToolStatus)


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_export_docx_real(tmp_path: Path):
    md = tmp_path / "in.md"
    md.write_text(GOLDEN_MD, encoding="utf-8")
    result = export_document(md, tmp_path, formats=["docx"])
    assert "docx" in result.outputs
    assert Path(result.outputs["docx"]).exists()
