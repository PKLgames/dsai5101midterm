"""Stage 2 pair-disjoint allocation and outcome-masked canonical rows."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .audit import _digest
from .canonicalize import canonical_outcome, canonical_pair, pair_id

PARTITIONS = ("fit", "validation", "test")
BANDS = ("1", "2-4", "5-9", "10+")
STRUCTURAL_FIELDS = (
    "train_row_id", "a_cat", "b_cat", "a_image", "b_image", "comparison_type",
    "gender", "pet_history", "pet_care", "pet_contact", "future_pet", "mbti_ie",
)
MANIFEST_SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("image_i", pa.string()), ("image_j", pa.string()),
    ("comparison_type", pa.string()), ("response_count", pa.int64()),
    ("response_band", pa.string()), ("allocation_stratum", pa.string()),
    ("partition", pa.string()), ("seed", pa.int64()),
])
ROW_SCHEMA = pa.schema(
    [(field, pa.string()) for field in STRUCTURAL_FIELDS]
    + [("image_i", pa.string()), ("image_j", pa.string()), ("pair_id", pa.string()),
       ("orientation", pa.int8()), ("partition", pa.string()),
       ("y_displayed", pa.int8()), ("y_canonical", pa.int8())]
)


def response_band(count: int) -> str:
    if count < 1:
        raise ValueError("A pair must have at least one response.")
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    return "10+"


def largest_remainder(count: int) -> tuple[int, int, int]:
    if count < 0:
        raise ValueError("Pair count cannot be negative.")
    products = [count * weight for weight in (70, 15, 15)]
    sizes = [product // 100 for product in products]
    remainder = count - sum(sizes)
    order = sorted(range(3), key=lambda index: (-(products[index] % 100), index))
    for index in order[:remainder]:
        sizes[index] += 1
    return sizes[0], sizes[1], sizes[2]


def collect_pairs(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        row_id = row["train_row_id"]
        if not row_id or row_id in seen_ids:
            raise ValueError(f"Missing or duplicate training row ID: {row_id!r}")
        seen_ids.add(row_id)
        i, j, _ = canonical_pair(row["a_image"], row["b_image"])
        key = pair_id(i, j)
        comparison = row["comparison_type"]
        if comparison not in {"within_cat", "between_cats"}:
            raise ValueError(f"Invalid comparison type for row {row_id}.")
        if key in pairs:
            if pairs[key]["comparison_type"] != comparison:
                raise ValueError(f"Pair has inconsistent comparison types: {key}")
            pairs[key]["response_count"] += 1
        else:
            pairs[key] = {
                "pair_id": key, "image_i": i, "image_j": j,
                "comparison_type": comparison, "response_count": 1,
            }
    if not pairs:
        raise ValueError("No non-quarantined training pairs are available.")
    return pairs


def allocate_pairs(pairs: dict[str, dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs.values():
        by_type[pair["comparison_type"]].append(pair)
    rng = random.Random(seed)
    assigned: list[dict[str, Any]] = []
    for comparison in sorted(by_type):
        group = by_type[comparison]
        counts = Counter(response_band(pair["response_count"]) for pair in group)
        collapse = any(counts[band] < 20 for band in BANDS)
        strata = ("all",) if collapse else BANDS
        for stratum in strata:
            members = sorted(
                (pair for pair in group if collapse or response_band(pair["response_count"]) == stratum),
                key=lambda pair: pair["pair_id"],
            )
            rng.shuffle(members)
            sizes = largest_remainder(len(members))
            offset = 0
            for partition, size in zip(PARTITIONS, sizes):
                for pair in members[offset:offset + size]:
                    assigned.append({
                        **pair, "response_band": response_band(pair["response_count"]),
                        "allocation_stratum": stratum, "partition": partition, "seed": seed,
                    })
                offset += size
    return sorted(assigned, key=lambda pair: pair["pair_id"])


def validate_manifest(
    pairs: dict[str, dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for record in records:
        key = record["pair_id"]
        if key in assignments or key not in pairs:
            raise ValueError(f"Existing split has a duplicate or unknown pair: {key}")
        expected = pairs[key]
        for field in ("image_i", "image_j", "comparison_type", "response_count"):
            if record[field] != expected[field]:
                raise ValueError(f"Existing split does not match data for {key}: {field}")
        if record["partition"] not in PARTITIONS:
            raise ValueError(f"Invalid split partition for {key}.")
        assignments[key] = record["partition"]
    if set(assignments) != set(pairs):
        raise ValueError("Existing split does not cover exactly the current non-quarantined pairs.")
    if set(assignments.values()) != set(PARTITIONS):
        raise ValueError("The split must contain fitting, validation, and test pairs.")
    return assignments


def canonical_rows(
    rows: list[dict[str, str]], assignments: dict[str, str], outcomes: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        i, j, orientation = canonical_pair(row["a_image"], row["b_image"])
        key = pair_id(i, j)
        partition = assignments[key]
        displayed: int | None = None
        canonical: int | None = None
        if partition != "test":
            choice = outcomes[row["train_row_id"]]
            canonical = canonical_outcome(choice, orientation)
            displayed = int(choice == "A")
        output.append({
            **row, "image_i": i, "image_j": j, "pair_id": key,
            "orientation": orientation, "partition": partition,
            "y_displayed": displayed, "y_canonical": canonical,
        })
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(STRUCTURAL_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Missing structural fields in {path}.")
        return [{field: row[field] for field in STRUCTURAL_FIELDS} for row in reader]


def _verify_stage1(root: Path) -> set[str]:
    with (root / "tables" / "data_audit.csv").open(encoding="utf-8", newline="") as stream:
        gates = [row for row in csv.DictReader(stream) if row["check"] == "fitting_gate"]
    if len(gates) != 1 or gates[0]["value"] != "no_blocking_audit_errors":
        raise ValueError("Stage 1 audit gate is missing or blocked.")
    with (root / "artifacts" / "cleaning_log.csv").open(encoding="utf-8", newline="") as stream:
        issues = list(csv.DictReader(stream))
    if any(issue["severity"] == "error" for issue in issues):
        raise ValueError("Stage 1 cleaning log contains blocking errors.")
    quarantine = {
        issue["row_id"] for issue in issues
        if issue["partition"] == "train" and issue["issue"] == "identical_image_self_comparison"
        and issue["action"] == "quarantine"
    }
    hashes = json.loads((root / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    if hashes["algorithm"] != "sha256":
        raise ValueError("Unsupported Stage 1 hash algorithm.")
    known_files = set(hashes["files"]) | {
        "dataset/images/" + name for name in hashes["unreadable_images"]
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in (root / "dataset" / "images").rglob("*") if path.is_file()
    } | {"dataset/train.csv", "dataset/test.csv"}
    if known_files != actual_files:
        raise ValueError("Raw-file inventory has changed since Stage 1.")
    for relative, expected in hashes["files"].items():
        path = root.joinpath(*relative.split("/"))
        digest, size = _digest(path)
        if digest != expected["sha256"] or size != expected["size_bytes"]:
            raise ValueError(f"Raw file changed since Stage 1: {relative}")
    return quarantine


def run_split(root: Path, config: dict[str, Any]) -> tuple[int, int, bool]:
    if not config.get("real_data_modeling_allowed"):
        raise ValueError("Stage 0 governance gate has not been opened.")
    if config.get("options", {}).get("known_cat_g3") is not False:
        raise ValueError("This Stage 2 implementation supports only the registered G3-off configuration.")
    split = config["split"]
    expected = {
        "unit": "unordered_photo_pair",
        "proportions_fit_validation_test": [0.70, 0.15, 0.15],
        "stratification": "comparison_type_by_pair_response_band",
        "bands": list(BANDS),
        "collapse_band_threshold_pairs": 20,
        "allocation": "largest_remainder",
        "allocation_tie_order": list(PARTITIONS),
    }
    if any(split.get(key) != value for key, value in expected.items()):
        raise ValueError("Registered split settings do not match this implementation.")
    seed = config["seeds"]["new_primary_split_if_needed"]
    if not isinstance(seed, int):
        raise ValueError("The registered new split seed must be an integer.")
    quarantine = _verify_stage1(root)
    all_rows = _read_csv(root / "dataset" / "train.csv")
    present_ids = {row["train_row_id"] for row in all_rows}
    if not quarantine <= present_ids:
        raise ValueError("The quarantine log refers to an unknown training row.")
    rows = [row for row in all_rows if row["train_row_id"] not in quarantine]
    pairs = collect_pairs(rows)
    manifest_path = root / "artifacts" / "split_manifest.parquet"
    rows_path = root / "artifacts" / "rows_canonical.parquet"
    if rows_path.exists() and not manifest_path.exists():
        raise ValueError("Canonical rows exist without a split manifest; refusing to replace them.")
    reused = manifest_path.exists()
    if reused:
        records = pq.read_table(manifest_path).to_pylist()
    else:
        records = allocate_pairs(pairs, seed)
    assignments = validate_manifest(pairs, records)

    dev_ids = {
        row["train_row_id"] for row in rows
        if assignments[pair_id(*canonical_pair(row["a_image"], row["b_image"])[:2])] != "test"
    }
    outcomes: dict[str, str] = {}
    with (root / "dataset" / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["train_row_id"] in dev_ids:
                outcomes[row["train_row_id"]] = row["choice"]
    if set(outcomes) != dev_ids:
        raise ValueError("Development outcomes do not cover the expected row IDs.")
    canonical = canonical_rows(rows, assignments, outcomes)
    table = pa.Table.from_pylist(canonical, schema=ROW_SCHEMA)
    hashes = json.loads((root / "artifacts" / "raw_hashes.json").read_text(encoding="utf-8"))
    raw_hash = hashes["files"]["dataset/train.csv"]
    if _digest(root / "dataset" / "train.csv") != (raw_hash["sha256"], raw_hash["size_bytes"]):
        raise ValueError("Training data changed while preparing the split.")
    if rows_path.exists() and not pq.read_table(rows_path).equals(table):
        raise ValueError("Existing canonical rows do not match the retained split and current data.")
    if not reused:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA), manifest_path)
    if not rows_path.exists():
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, rows_path)
    return len(rows), len(pairs), reused
