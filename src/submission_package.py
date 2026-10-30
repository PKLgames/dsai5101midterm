"""Restricted course-only package with portable cached-feature reproduction inputs."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .development import _hash
from .reproduce import verify_history

DIRECTORIES = ("src", "tests", "legacy", "features", "models", "tables", "artifacts", "figures")
ROOT_FILES = (
    "README.md", "REPRODUCTION.md", "references.csv", "protocol.md", "decisions.yaml",
    "project-plan-source.txt", "claims.csv", "claims_final.csv", "contributions.csv",
    "requirements.txt", "requirements-stage2.txt", "requirements-stage3.txt",
    "requirements-stage4.txt", "requirements-stage5.txt", "requirements-stage7.txt",
    "requirements-stage11.txt", "requirements-y-gpu.txt",
    "requirements-lock-cpu.txt", "requirements-lock-y-gpu.txt", "Group_19_Report.pdf",
    "dataset/train.csv", "dataset/test.csv",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".jpg", ".jpeg", ".png", ".pyc"}


def included(relative: Path) -> bool:
    parts = relative.parts
    if any(part in {".git", ".venv", ".cache", "__pycache__", "review_panels"} for part in parts):
        return False
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if any(part.startswith(".env") or part.lower() in {"credentials", "secrets"} for part in parts):
        return False
    if relative.as_posix() in {"artifacts/stage11_package_provenance.json", "artifacts/archive_reproduction.json"}:
        return False
    return True


def package(root: Path) -> dict[str, Any]:
    archive_path = root / "Group_19_Code.zip"
    provenance = root / "artifacts/stage11_package_provenance.json"
    if archive_path.exists() or provenance.exists():
        raise FileExistsError("Code archive or package provenance already exists.")
    verify_history(root)
    decisions = json.loads((root / "artifacts/stage11_decisions.json").read_text(encoding="utf-8"))
    if decisions.get("course_only_bundle") is not True:
        raise ValueError("This packaging route requires user authorization for course-only cached data.")
    reproduction = json.loads((root / "artifacts/reproduction_statistical.json").read_text(encoding="utf-8"))
    report = json.loads((root / "artifacts/stage11_report_provenance.json").read_text(encoding="utf-8"))
    if reproduction["status"] != "passed" or _hash(root / "Group_19_Report.pdf") != report["report_sha256"]:
        raise ValueError("Reproduction gate or report identity check failed.")
    pdf = PdfReader(root / "Group_19_Report.pdf")
    if len(pdf.pages) != report["total_pages"] or not any(
        "TEAM MUST EDIT BEFORE SUBMISSION" in (page.extract_text() or "") for page in pdf.pages
    ):
        raise ValueError("Draft report lost its required contribution-placeholder warning.")
    files: dict[str, Path] = {}
    for name in ROOT_FILES:
        path = root.joinpath(*name.split("/"))
        if not path.is_file():
            raise FileNotFoundError(f"Required code-package input is missing: {name}")
        files[name] = path
    for directory in DIRECTORIES:
        for path in (root / directory).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Do not archive symlinks: {path}")
            if path.is_file():
                relative = path.relative_to(root)
                if included(relative):
                    files[relative.as_posix()] = path
    entries = [{"path": name, "bytes": path.stat().st_size, "sha256": _hash(path)}
               for name, path in sorted(files.items())]
    manifest = {
        "group": "19", "purpose": "private course education/research reproduction",
        "report_status": report["status"],
        "data_restriction": decisions["data_restriction"],
        "excluded": ["raw photographs and review panels", "pretrained weights", "environments", "Git", "caches", "secrets"],
        "entries": entries,
    }
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Code archive CRC verification failed.")
        expected_names = set(files) | {"PACKAGE_MANIFEST.json"}
        if set(archive.namelist()) != expected_names or len(archive.namelist()) != len(expected_names):
            raise ValueError("Code archive entry inventory is inconsistent.")
        for entry in entries:
            import hashlib
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise ValueError(f"Archived input changed: {entry['path']}")
    result = {
        "status": "local_draft_package_created_not_submitted",
        "archive": archive_path.name, "archive_sha256": _hash(archive_path),
        "archive_bytes": archive_path.stat().st_size, "payload_files": len(files),
        "manifest": manifest, "raw_photographs_included": False, "pretrained_weights_included": False,
        "contribution_placeholder_requires_manual_edit": True,
        "course_chapter7_reference_requires_completion": True,
        "submission_channel_verified": False, "uploaded": False,
    }
    provenance.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        report = package(args.root.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Submission packaging failed: {exc}", file=sys.stderr)
        return 2
    print(f"Created {report['archive']}: {report['payload_files']} payload files; private course use only; no upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
