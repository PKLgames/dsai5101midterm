"""Stage 1 raw-data audit. No outcomes are used for model selection."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


CONTEXT = {
    "pet_history": {"current", "past", "never"},
    "pet_care": {"never", "occasionally", "past_regular", "current_regular"},
    "pet_contact": {"never", "rarely", "sometimes", "often", "daily"},
    "future_pet": {"yes", "no", "unsure"},
    "mbti_ie": {"I", "E", "unknown"},
}
BASE_FIELDS = {
    "a_cat", "b_cat", "a_image", "b_image", "choice", "comparison_type",
    "gender", *CONTEXT,
}
AUDIT_FIELDS = ("partition", "check", "value", "count", "status", "details")
LOG_FIELDS = ("partition", "row_id", "issue", "severity", "action", "details")
COVERAGE_FIELDS = (
    "test_row_id", "overlap_category", "missing_a_image", "missing_b_image",
    "fitting_component_support", "feature_availability", "quarantine_reason",
)


@dataclass
class AuditResult:
    metrics: list[dict[str, Any]]
    issues: list[dict[str, str]]
    coverage: list[dict[str, str]]
    hashes: dict[str, Any]

    @property
    def blocked(self) -> bool:
        return any(issue["severity"] == "error" for issue in self.issues)


def _digest(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while being hashed: {path}")
    return digest.hexdigest(), after.st_size


def _read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError(f"{path}: missing CSV header")
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            rows.append(row)
    return rows, fields


def _image_key(value: str) -> str | None:
    if not value or "\\" in value or ":" in value:
        return None
    parts = value.split("/")
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return None
    path = PurePosixPath(value)
    if path.is_absolute():
        return None
    return path.as_posix()


def _pair(row: dict[str, str]) -> tuple[str, str]:
    first, second = sorted((row["a_image"], row["b_image"]))
    return first, second


def _metric(
    metrics: list[dict[str, Any]], partition: str, check: str, value: str,
    count: int, status: str = "info", details: str = "",
) -> None:
    metrics.append({
        "partition": partition, "check": check, "value": value,
        "count": count, "status": status, "details": details,
    })


def _issue(
    issues: list[dict[str, str]], partition: str, row_id: str, issue: str,
    severity: str, action: str, details: str,
) -> None:
    issues.append({
        "partition": partition, "row_id": row_id, "issue": issue,
        "severity": severity, "action": action, "details": details,
    })


def audit_dataset(root: Path) -> AuditResult:
    dataset = root / "dataset"
    images = dataset / "images"
    if not images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images}")

    metrics: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    coverage: list[dict[str, str]] = []
    hashes: dict[str, Any] = {
        "algorithm": "sha256", "files": {}, "unreadable_images": {},
        "duplicate_image_contents": [],
    }

    partitions: dict[str, list[dict[str, str]]] = {}
    headers: dict[str, list[str]] = {}
    for partition, id_field in (("train", "train_row_id"), ("test", "test_row_id")):
        path = dataset / f"{partition}.csv"
        required = BASE_FIELDS | {id_field}
        if partition == "test":
            required |= {"prob_a"}
        digest, size = _digest(path)
        rows, fields = _read_csv(path, required)
        if _digest(path) != (digest, size):
            raise RuntimeError(f"CSV changed while being audited: {path}")
        hashes["files"][f"dataset/{partition}.csv"] = {"sha256": digest, "size_bytes": size}
        partitions[partition], headers[partition] = rows, fields
        _metric(metrics, partition, "row_count", "all", len(rows))
        expected = 24896 if partition == "train" else 6224
        if len(rows) != expected:
            _issue(issues, partition, "", "documented_row_count_mismatch", "warning",
                   "review", f"Expected {expected}; observed {len(rows)}.")

    image_hashes: dict[str, list[str]] = defaultdict(list)
    image_files: set[str] = set()
    for path in sorted(images.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in image tree requires review: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(images).as_posix()
        image_files.add(relative)
        try:
            digest, size = _digest(path)
        except OSError as exc:
            hashes["unreadable_images"][relative] = str(exc)
            _issue(issues, "images", relative, "unreadable_image_file", "warning",
                   "feature_unavailable", str(exc))
            continue
        hashes["files"][f"dataset/images/{relative}"] = {"sha256": digest, "size_bytes": size}
        image_hashes[digest].append(relative)

    for digest, paths in sorted(image_hashes.items()):
        if len(paths) > 1:
            hashes["duplicate_image_contents"].append({"sha256": digest, "paths": paths})
            _issue(issues, "images", "", "duplicate_image_content", "warning",
                   "retain_distinct_paths", json.dumps(paths, ensure_ascii=False))
    _metric(metrics, "images", "file_count", "all", len(image_files))
    _metric(metrics, "images", "duplicate_content_groups", "sha256", len(hashes["duplicate_image_contents"]))
    _metric(metrics, "images", "unreadable_files", "all", len(hashes["unreadable_images"]))
    if len(image_files) != 1558:
        _issue(issues, "images", "", "documented_image_count_mismatch", "warning",
               "review", f"Expected 1558; observed {len(image_files)}.")

    train_cats: set[str] = set()
    train_images: set[str] = set()
    train_pairs: set[tuple[str, str]] = set()
    valid_paths: dict[tuple[str, int, str], str | None] = {}
    for partition in ("train", "test"):
        rows = partitions[partition]
        id_field = f"{partition}_row_id"
        ids = Counter(row[id_field] for row in rows)
        for row_id, count in sorted(ids.items()):
            if not row_id:
                _issue(issues, partition, "", "missing_row_id", "error", "block_fitting",
                       f"{count} rows have a blank ID.")
            elif count > 1:
                _issue(issues, partition, row_id, "duplicate_row_id", "error",
                       "block_fitting", f"ID occurs {count} times.")
        _metric(metrics, partition, "unique_row_ids", "nonempty", len(ids) - ("" in ids))
        _metric(metrics, partition, "duplicate_row_id_values", "all",
                sum(1 for key, count in ids.items() if key and count > 1),
                "error" if any(key and count > 1 for key, count in ids.items()) else "ok")

        duplicates: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for row in rows:
            duplicates[tuple(row[field] for field in headers[partition] if field != id_field)].append(row[id_field])
        repeated = [group for group in duplicates.values() if len(group) > 1]
        _metric(metrics, partition, "identical_response_groups_excluding_id", "all", len(repeated))
        for group in repeated:
            _issue(issues, partition, "", "identical_looking_responses", "info",
                   "retain_all", json.dumps(group, ensure_ascii=False))

        for field in ("gender", *CONTEXT):
            values = Counter(row[field] for row in rows)
            for value, count in sorted(values.items()):
                status = "missing" if value == "" else (
                    "documented_unknown" if field == "mbti_ie" and value == "unknown" else (
                        "unexpected" if field in CONTEXT and value not in CONTEXT[field] else "observed"
                    )
                )
                _metric(metrics, partition, f"context_{field}", value or "<missing>", count, status)
                if status == "unexpected":
                    _issue(issues, partition, "", "unexpected_context_code", "error",
                           "review_before_fitting", f"{field}={value!r}; count={count}")

        for field in ("choice", "comparison_type", *(("prob_a",) if partition == "test" else ())):
            values = Counter(
                ("<nonblank>" if row[field] else "<blank>")
                if partition == "test" and field in {"choice", "prob_a"} else row[field]
                for row in rows
            )
            for value, count in sorted(values.items()):
                _metric(metrics, partition, f"observed_{field}", value or "<missing>", count)

        for index, row in enumerate(rows):
            row_id = row[id_field]
            row_errors = False
            for side in ("a", "b"):
                cat, image = row[f"{side}_cat"], row[f"{side}_image"]
                key = _image_key(image)
                valid_paths[(partition, index, side)] = key
                if not cat:
                    row_errors = True
                    _issue(issues, partition, row_id, "missing_cat_id", "error",
                           "block_fitting", f"{side}_cat is blank.")
                if key is None:
                    row_errors = True
                    _issue(issues, partition, row_id, "invalid_image_path", "error",
                           "block_fitting", f"{side}_image={image!r}")
                elif cat and PurePosixPath(key).parts[0] != cat:
                    row_errors = True
                    _issue(issues, partition, row_id, "cat_image_folder_mismatch", "error",
                           "block_fitting", f"{side}_cat={cat!r}; {side}_image={image!r}")
                if key is not None and key not in image_files:
                    _issue(issues, partition, row_id, "missing_image_file", "warning",
                           "feature_unavailable", f"{side}_image={image!r}")

            comparison = row["comparison_type"]
            if comparison not in {"within_cat", "between_cats"}:
                row_errors = True
                _issue(issues, partition, row_id, "invalid_comparison_type", "error",
                       "block_fitting", repr(comparison))
            elif row["a_cat"] and row["b_cat"] and (
                (comparison == "within_cat") != (row["a_cat"] == row["b_cat"])
            ):
                row_errors = True
                _issue(issues, partition, row_id, "comparison_cat_mismatch", "error",
                       "block_fitting", f"comparison_type={comparison}")

            if partition == "train" and row["choice"] not in {"A", "B"}:
                row_errors = True
                _issue(issues, partition, row_id, "invalid_training_choice", "error",
                       "block_fitting", f"choice={row['choice']!r}")
            if partition == "test" and (row["choice"] != "" or row["prob_a"] != ""):
                _issue(issues, partition, row_id, "nonblank_official_test_output", "error",
                       "review_before_fitting", "Expected blank choice and prob_a.")

            self_comparison = bool(row["a_image"]) and row["a_image"] == row["b_image"]
            if self_comparison:
                _issue(issues, partition, row_id, "identical_image_self_comparison",
                       "warning", "quarantine", row["a_image"])
            if partition == "train" and not row_errors and not self_comparison:
                train_cats.update((row["a_cat"], row["b_cat"]))
                train_images.update((row["a_image"], row["b_image"]))
                train_pairs.add(_pair(row))

    for index, row in enumerate(partitions["test"]):
        if row["a_cat"] not in train_cats or row["b_cat"] not in train_cats:
            category = "at_least_one_unseen_cat"
        elif row["a_image"] not in train_images or row["b_image"] not in train_images:
            category = "at_least_one_unseen_photo"
        elif _pair(row) not in train_pairs:
            category = "unseen_pair_of_known_photos"
        else:
            category = "previously_observed_pair"
        a = valid_paths[("test", index, "a")]
        b = valid_paths[("test", index, "b")]
        coverage.append({
            "test_row_id": row["test_row_id"],
            "overlap_category": category,
            "missing_a_image": str(a is None or a not in image_files).lower(),
            "missing_b_image": str(b is None or b not in image_files).lower(),
            "fitting_component_support": "not_available_before_split",
            "feature_availability": "not_available_before_feature_integration",
            "quarantine_reason": (
                "identical_image_self_comparison"
                if row["a_image"] and row["a_image"] == row["b_image"] else ""
            ),
        })

    category_counts = Counter(row["overlap_category"] for row in coverage)
    for category in (
        "at_least_one_unseen_cat", "at_least_one_unseen_photo",
        "unseen_pair_of_known_photos", "previously_observed_pair",
    ):
        _metric(metrics, "test", "official_overlap_category", category, category_counts[category],
                details="Compared with valid non-self training rows; not a performance estimate.")
    for partition in ("train", "test", "images"):
        counts = Counter(issue["issue"] for issue in issues if issue["partition"] == partition)
        for issue, count in sorted(counts.items()):
            _metric(metrics, partition, "issue_count", issue, count)
    _metric(metrics, "audit", "fitting_component_support", "not_available_before_split", 0,
            "deferred", "Stage 2 split and Stage 4 graph required.")
    _metric(metrics, "audit", "feature_availability", "not_available_before_feature_integration", 0,
            "deferred", "Stage 3 feature integration required.")
    result = AuditResult(metrics, issues, coverage, hashes)
    _metric(metrics, "audit", "fitting_gate", "blocked" if result.blocked else "no_blocking_audit_errors",
            sum(issue["severity"] == "error" for issue in issues),
            "error" if result.blocked else "ok",
            "Stage 0 governance gate remains separate.")
    return result


def _csv_text(fields: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_audit(root: Path, result: AuditResult, overwrite: bool = False) -> list[Path]:
    outputs = {
        root / "tables" / "data_audit.csv": _csv_text(AUDIT_FIELDS, result.metrics),
        root / "tables" / "official_test_coverage.csv": _csv_text(COVERAGE_FIELDS, result.coverage),
        root / "artifacts" / "cleaning_log.csv": _csv_text(LOG_FIELDS, result.issues),
        root / "artifacts" / "raw_hashes.json": json.dumps(result.hashes, indent=2, ensure_ascii=False) + "\n",
    }
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Audit outputs already exist; use --overwrite to replace: " +
                              ", ".join(str(path) for path in existing))
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    return list(outputs)
