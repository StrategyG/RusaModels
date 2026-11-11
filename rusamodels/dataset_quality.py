"""Find near-duplicate records and cross-split contamination in a dataset release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


_BANDS = 8
_BITS_PER_BAND = 8
_MAX_HAMMING_DISTANCE = 7
_TOKEN = re.compile(r"\w+", re.UNICODE)


class DatasetQualityError(ValueError):
    """Raised when dataset quality analysis cannot be completed safely."""


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _text_values(item)]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            if key not in {"id", "group_id"}
            for text in _text_values(item)
        ]
    return []


def _simhash64(record: dict[str, Any]) -> int:
    tokens = _TOKEN.findall(" ".join(_text_values(record)).casefold())
    if not tokens:
        return 0
    if len(tokens) >= 3:
        features = {
            " ".join(tokens[index:index + 3])
            for index in range(len(tokens) - 2)
        }
    else:
        features = set(tokens)
    weights = [0] * 64
    for feature in features:
        digest = hashlib.blake2b(
            feature.encode(), digest_size=8, person=b"RusaData"
        ).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, weight in enumerate(weights) if weight >= 0)


def _bands(simhash: int) -> tuple[int, ...]:
    mask = (1 << _BITS_PER_BAND) - 1
    return tuple(
        (simhash >> (index * _BITS_PER_BAND)) & mask
        for index in range(_BANDS)
    )


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE record_groups (
            record_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL
        );
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            group_id TEXT NOT NULL,
            simhash_hex TEXT NOT NULL
        );
        CREATE TABLE buckets (
            band_index INTEGER NOT NULL,
            band_value INTEGER NOT NULL,
            record_id TEXT NOT NULL,
            PRIMARY KEY (band_index, band_value, record_id)
        );
        CREATE INDEX ix_buckets_lookup ON buckets(band_index, band_value);
        CREATE TABLE pairs (
            record_id_a TEXT NOT NULL,
            record_id_b TEXT NOT NULL,
            split_a TEXT NOT NULL,
            split_b TEXT NOT NULL,
            group_id_a TEXT NOT NULL,
            group_id_b TEXT NOT NULL,
            hamming_distance INTEGER NOT NULL,
            PRIMARY KEY (record_id_a, record_id_b)
        );
        """
    )


def _load_group_provenance(
    connection: sqlite3.Connection,
    provenance_path: Path,
) -> None:
    with provenance_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            item = json.loads(line)
            record_id = item.get("record_id")
            group_id = item.get("group_id")
            if not isinstance(record_id, str) or not isinstance(group_id, str):
                raise DatasetQualityError(
                    f"provenance.jsonl:{line_number}: record_id/group_id is invalid"
                )
            existing = connection.execute(
                "SELECT group_id FROM record_groups WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if existing and existing[0] != group_id:
                raise DatasetQualityError(
                    f"record {record_id} has inconsistent group provenance"
                )
            connection.execute(
                "INSERT OR IGNORE INTO record_groups(record_id, group_id) VALUES (?, ?)",
                (record_id, group_id),
            )


def find_near_duplicates(
    release_path: str | Path,
    output_path: str | Path,
    *,
    max_hamming_distance: int = 7,
    max_candidates_per_record: int = 10000,
) -> dict[str, Any]:
    """Write a near-duplicate report and count cross-split contamination pairs.

    Eight 8-bit LSH bands guarantee candidate retrieval for SimHash distances
    up to seven bits. This is a candidate filter, not a semantic similarity
    proof; every candidate is checked using exact Hamming distance.
    """
    if type(max_hamming_distance) is not int or not 0 <= max_hamming_distance <= _MAX_HAMMING_DISTANCE:
        raise DatasetQualityError(
            f"max_hamming_distance must be between 0 and {_MAX_HAMMING_DISTANCE}"
        )
    if type(max_candidates_per_record) is not int or max_candidates_per_record < 1:
        raise DatasetQualityError("max_candidates_per_record must be a positive integer")

    root = Path(release_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if root == output or root in output.parents:
        raise DatasetQualityError("near-duplicate report must be outside the immutable dataset release")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        verification = verify_dataset_release(root)
    except (DatasetVerificationError, OSError, UnicodeError) as exc:
        raise DatasetQualityError(f"dataset release verification failed: {exc}") from exc

    temporary_report: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="rusamodels-quality-") as temporary_dir:
            database_path = Path(temporary_dir) / "quality.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                _initialize(connection)
                _load_group_provenance(connection, root / "provenance.jsonl")
                connection.commit()
                candidate_comparisons = 0
                records_scanned = 0
                clauses = " OR ".join(
                    "(band_index = ? AND band_value = ?)" for _ in range(_BANDS)
                )
                for split in ("train", "validation", "test"):
                    split_path = root / f"{split}.jsonl"
                    with split_path.open("r", encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            try:
                                record = json.loads(line)
                                record_id = record["id"]
                            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                                raise DatasetQualityError(
                                    f"{split}.jsonl:{line_number}: invalid record"
                                ) from exc
                            group_row = connection.execute(
                                "SELECT group_id FROM record_groups WHERE record_id = ?",
                                (record_id,),
                            ).fetchone()
                            if group_row is None:
                                raise DatasetQualityError(
                                    f"{split}.jsonl:{line_number}: missing group provenance"
                                )
                            group_id = group_row[0]
                            simhash = _simhash64(record)
                            band_values = _bands(simhash)
                            parameters = tuple(
                                value
                                for index, band_value in enumerate(band_values)
                                for value in (index, band_value)
                            )
                            candidates = [
                                row[0]
                                for row in connection.execute(
                                    f"SELECT DISTINCT record_id FROM buckets WHERE {clauses}",
                                    parameters,
                                )
                            ]
                            if len(candidates) > max_candidates_per_record:
                                raise DatasetQualityError(
                                    f"{split}.jsonl:{line_number}: candidate set exceeds "
                                    f"max_candidates_per_record={max_candidates_per_record}"
                                )
                            for candidate_id in candidates:
                                candidate = connection.execute(
                                    "SELECT split, group_id, simhash_hex FROM records "
                                    "WHERE record_id = ?",
                                    (candidate_id,),
                                ).fetchone()
                                if candidate is None or candidate[1] == group_id:
                                    continue
                                candidate_comparisons += 1
                                distance = (
                                    simhash ^ int(candidate[2], 16)
                                ).bit_count()
                                if distance > max_hamming_distance:
                                    continue
                                if candidate_id < record_id:
                                    pair = (
                                        candidate_id, record_id, candidate[0], split,
                                        candidate[1], group_id, distance,
                                    )
                                else:
                                    pair = (
                                        record_id, candidate_id, split, candidate[0],
                                        group_id, candidate[1], distance,
                                    )
                                connection.execute(
                                    """
                                    INSERT OR IGNORE INTO pairs(
                                        record_id_a, record_id_b, split_a, split_b,
                                        group_id_a, group_id_b, hamming_distance
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    pair,
                                )
                            connection.execute(
                                "INSERT INTO records(record_id, split, group_id, simhash_hex) "
                                "VALUES (?, ?, ?, ?)",
                                (record_id, split, group_id, f"{simhash:016x}"),
                            )
                            connection.executemany(
                                "INSERT INTO buckets(band_index, band_value, record_id) "
                                "VALUES (?, ?, ?)",
                                [
                                    (index, value, record_id)
                                    for index, value in enumerate(band_values)
                                ],
                            )
                            records_scanned += 1
                connection.commit()
                temporary_fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
                )
                os.close(temporary_fd)
                temporary_report = Path(temporary_name)
                near_duplicate_pairs = 0
                cross_split_pairs = 0
                with temporary_report.open("w", encoding="utf-8", newline="\n") as report:
                    rows = connection.execute(
                        """
                        SELECT record_id_a, record_id_b, split_a, split_b,
                               group_id_a, group_id_b, hamming_distance
                        FROM pairs
                        ORDER BY hamming_distance, record_id_a, record_id_b
                        """
                    )
                    for row in rows:
                        item = {
                            "record_id_a": row[0],
                            "record_id_b": row[1],
                            "split_a": row[2],
                            "split_b": row[3],
                            "group_id_a": row[4],
                            "group_id_b": row[5],
                            "hamming_distance": row[6],
                        }
                        report.write(json.dumps(item, sort_keys=True) + "\n")
                        near_duplicate_pairs += 1
                        if row[2] != row[3]:
                            cross_split_pairs += 1
            finally:
                connection.close()
        os.replace(temporary_report, output)
        temporary_report = None
        return {
            "dataset_id": verification["dataset_id"],
            "dataset_version": verification["dataset_version"],
            "records_scanned": records_scanned,
            "candidate_comparisons": candidate_comparisons,
            "near_duplicate_pairs": near_duplicate_pairs,
            "cross_split_pairs": cross_split_pairs,
            "max_hamming_distance": max_hamming_distance,
            "algorithm": "simhash64_word_trigram_lsh_8x8",
            "report_path": str(output),
        }
    except (OSError, sqlite3.Error, UnicodeError) as exc:
        raise DatasetQualityError(f"near-duplicate analysis failed: {exc}") from exc
    finally:
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path, help="Verified dataset release directory")
    parser.add_argument("--output", required=True, type=Path, help="Report path outside the immutable release")
    parser.add_argument("--max-hamming-distance", type=int, default=7)
    parser.add_argument("--max-candidates-per-record", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        result = find_near_duplicates(
            args.release,
            args.output,
            max_hamming_distance=args.max_hamming_distance,
            max_candidates_per_record=args.max_candidates_per_record,
        )
    except (DatasetQualityError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["cross_split_pairs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
