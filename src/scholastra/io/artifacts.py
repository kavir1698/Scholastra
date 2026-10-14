"""Project workspace and artifact IO (manifest, evidence, drafts, checkpoints)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholastra.models.schemas import (
    Brief,
    DocumentContract,
    EditorialFinding,
    Outline,
    QualityReport,
    ScopeProposal,
    SectionDraft,
    SectionEvidence,
    Source,
)
from scholastra.utils import slugify, stable_id

_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def validate_project_id(project_id: str) -> str:
    """Reject ids that are not a single safe path component."""
    pid = (project_id or "").strip()
    if not pid or pid in {".", ".."} or ".." in pid or not _PROJECT_ID_RE.fullmatch(pid):
        raise ValueError(
            f"invalid project id {project_id!r}: expected a single path component of "
            "letters, digits, '.', '_' or '-'"
        )
    return pid


def new_project_id(topic: str, *, doc_type: str = "article") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(topic, max_len=32)}-{doc_type}-{stamp}"


@dataclass
class Workspace:
    """A single project's on-disk workspace."""

    root: Path
    project_id: str
    _manifest: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def dir(self) -> Path:
        return self.root / self.project_id

    @property
    def artifacts_dir(self) -> Path:
        return self.dir / "artifacts"

    @property
    def evidence_dir(self) -> Path:
        return self.artifacts_dir / "evidence"

    @property
    def drafts_dir(self) -> Path:
        return self.artifacts_dir / "drafts"

    @property
    def checkpoint_path(self) -> Path:
        return self.dir / "checkpoints.sqlite"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def run_log_path(self) -> Path:
        return self.dir / "run.log.jsonl"

    def ensure(self) -> Workspace:
        validate_project_id(self.project_id)
        base = self.root.resolve()
        target = self.dir.resolve()
        if target != base and not target.is_relative_to(base):
            raise ValueError(f"project directory escapes the workspace root: {target}")
        for directory in (self.dir, self.artifacts_dir, self.evidence_dir, self.drafts_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    # -- generic IO --------------------------------------------------------
    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def _write_target(self, rel: str) -> Path:
        target = self.path(rel)
        resolved = target.resolve()
        base = self.dir.resolve()
        if resolved != base and not resolved.is_relative_to(base):
            raise ValueError(f"refusing to write outside the project workspace: {rel}")
        return target

    def write_text(self, rel: str, text: str) -> Path:
        target = self._write_target(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def read_text(self, rel: str) -> str:
        return self.path(rel).read_text(encoding="utf-8")

    def write_json(self, rel: str, data: Any) -> Path:
        target = self._write_target(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def read_json(self, rel: str, default: Any = None) -> Any:
        target = self.path(rel)
        if not target.exists():
            return default
        return json.loads(target.read_text(encoding="utf-8"))

    # -- typed artifacts ---------------------------------------------------
    def save_scope(self, scope: ScopeProposal) -> Path:
        return self.write_json("artifacts/scope.json", scope.model_dump())

    def load_scope(self) -> ScopeProposal | None:
        data = self.read_json("artifacts/scope.json")
        return ScopeProposal.model_validate(data) if data else None

    def save_outline(self, outline: Outline) -> Path:
        return self.write_json("artifacts/outline.json", outline.model_dump())

    def load_outline(self) -> Outline | None:
        data = self.read_json("artifacts/outline.json")
        return Outline.model_validate(data) if data else None

    def save_evidence(self, evidence: SectionEvidence) -> Path:
        return self.write_json(
            f"artifacts/evidence/{evidence.section_id}.json", evidence.model_dump()
        )

    def save_draft(self, draft: SectionDraft) -> Path:
        return self.write_text(f"artifacts/drafts/{draft.section_id}.md", draft.markdown)

    def save_sources(self, sources: dict[str, Source]) -> Path:
        return self.write_json(
            "artifacts/sources.json", {sid: s.model_dump() for sid, s in sources.items()}
        )

    def load_sources(self) -> dict[str, Source]:
        data = self.read_json("artifacts/sources.json", default={}) or {}
        return {sid: Source.model_validate(entry) for sid, entry in data.items()}

    def save_brief(self, brief: Brief) -> Path:
        return self.write_json("artifacts/brief.json", brief.model_dump())

    def load_brief(self) -> Brief | None:
        data = self.read_json("artifacts/brief.json")
        return Brief.model_validate(data) if data else None

    def save_contract(self, contract: DocumentContract) -> Path:
        return self.write_json("artifacts/contract.json", contract.model_dump())

    def load_contract(self) -> DocumentContract | None:
        data = self.read_json("artifacts/contract.json")
        return DocumentContract.model_validate(data) if data else None

    def save_editorial(self, findings: list[EditorialFinding]) -> Path:
        return self.write_json(
            "artifacts/editorial.json", [f.model_dump() for f in findings]
        )

    def save_quality_report(self, report: QualityReport) -> Path:
        return self.write_json("artifacts/quality_report.json", report.model_dump())

    # -- manifest ----------------------------------------------------------
    def load_manifest(self) -> dict[str, Any]:
        if not self._manifest:
            self._manifest = self.read_json("manifest.json", default={}) or {}
        return self._manifest

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        manifest = self.load_manifest()
        manifest.update(fields)
        manifest.setdefault("project_id", self.project_id)
        manifest.setdefault("created_at", _utcnow())
        manifest["updated_at"] = _utcnow()
        artifacts = manifest.setdefault("artifacts", {})
        for rel in (
            "artifacts/brief.json",
            "artifacts/scope.json",
            "artifacts/outline.json",
            "artifacts/contract.json",
            "artifacts/editorial.json",
            "artifacts/quality_report.json",
            "artifacts/sources.json",
            "artifacts/final.md",
            "artifacts/refs.bib",
        ):
            if self.path(rel).exists():
                artifacts[Path(rel).stem] = rel
        self.write_json("manifest.json", manifest)
        self._manifest = manifest
        return manifest

    def project_dir_for_id(self, project_id: str) -> Path:
        return self.root / project_id


def open_workspace(root: str | Path, project_id: str) -> Workspace:
    return Workspace(root=Path(root), project_id=project_id).ensure()


def resolve_workspace_root(settings_root: str | Path) -> Path:
    root = Path(settings_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root


def make_project_id(topic: str, doc_type: str = "article") -> str:
    return new_project_id(topic, doc_type=doc_type) or stable_id(topic)
