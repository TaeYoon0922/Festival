"""Deterministically choose a diverse 20-document parsing sample."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Callable


GROUPS = ("periodic", "exchange", "major", "holding")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _unicode_child(parent: Path, name: str) -> Path:
    direct = parent / name
    if direct.exists():
        return direct
    normalized = unicodedata.normalize("NFC", name)
    for child in parent.iterdir():
        if unicodedata.normalize("NFC", child.name) == normalized:
            return child
    return direct


def resolve_unicode_path(base: Path, relative: str) -> Path:
    current = base
    for part in Path(relative).parts:
        current = _unicode_child(current, part)
    return current


def resolve_primary_xml(corpus_dir: Path, row: dict[str, Any]) -> Path:
    folder = resolve_unicode_path(corpus_dir, str(row["file_path"]))
    receipt = str(row["rcept_no"])
    expected = folder / f"{receipt}.xml"
    if expected.is_file():
        return expected
    candidates = sorted(folder.glob("*.xml")) if folder.is_dir() else []
    exact = [path for path in candidates if path.stem == receipt]
    if exact:
        return exact[0]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_size)
    raise FileNotFoundError(f"No XML found for {row.get('doc_id')}: {folder}")


def _prepare_candidates(
    manifest: list[dict[str, Any]], corpus_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    candidates = {group: [] for group in GROUPS}
    for original in manifest:
        group = original.get("doc_group")
        if (
            group not in candidates
            or original.get("is_correction")
            or original.get("file_format") != "xml"
        ):
            continue
        row = dict(original)
        source = resolve_primary_xml(corpus_dir, row)
        row["source_path"] = str(source.relative_to(corpus_dir))
        row["source_size"] = source.stat().st_size
        candidates[str(group)].append(row)

    for rows in candidates.values():
        rows.sort(
            key=lambda row: (
                int(row["source_size"]),
                str(row.get("rcept_dt", "")),
                str(row["rcept_no"]),
            )
        )
    return candidates


def _take_first(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool] = lambda _: True,
    prefer_new_company: bool = True,
) -> bool:
    selected_ids = {row["doc_id"] for row in selected}
    used_companies = {row["corp_name"] for row in selected}
    matches = [
        row
        for row in candidates
        if row["doc_id"] not in selected_ids and predicate(row)
    ]
    if prefer_new_company:
        distinct = [row for row in matches if row["corp_name"] not in used_companies]
        if distinct:
            matches = distinct
    if not matches:
        return False
    selected.append(matches[0])
    return True


def select_sample_documents(
    manifest_path: Path,
    corpus_dir: Path,
    per_group: int = 5,
) -> list[dict[str, Any]]:
    if per_group != 5:
        raise ValueError("The validated pilot currently requires exactly 5 documents per group")

    candidates = _prepare_candidates(load_manifest(manifest_path), corpus_dir)
    selected_by_group: dict[str, list[dict[str, Any]]] = {
        group: [] for group in GROUPS
    }

    periodic = selected_by_group["periodic"]
    samsung_annual = sorted(
        [
            row
            for row in candidates["periodic"]
            if row["corp_name"] == "삼성전자" and row["doc_subtype"] == "annual"
        ],
        key=lambda row: (str(row["rcept_dt"]), str(row["rcept_no"])),
    )
    if samsung_annual:
        periodic.append(samsung_annual[0])
    for subtype in ("half", "quarter", "annual", "quarter"):
        _take_first(
            candidates["periodic"],
            periodic,
            predicate=lambda row, subtype=subtype: row["doc_subtype"] == subtype,
        )

    exchange = selected_by_group["exchange"]
    for subtype in (
        "단일판매공급계약체결",
        "신규시설투자등",
        "투자판단관련주요경영사항",
        "단일판매공급계약해지",
    ):
        _take_first(
            candidates["exchange"],
            exchange,
            predicate=lambda row, subtype=subtype: row["doc_subtype"] == subtype,
        )

    major = selected_by_group["major"]
    for keyword in (
        "유상증자결정",
        "회사합병결정",
        "전환사채권발행결정",
        "자기주식처분결정",
        "회사분할결정",
    ):
        _take_first(
            candidates["major"],
            major,
            predicate=lambda row, keyword=keyword: keyword
            in str(row.get("report_nm", "")),
        )

    holding = selected_by_group["holding"]
    while len(holding) < per_group:
        if not _take_first(candidates["holding"], holding):
            break

    for group in GROUPS:
        while len(selected_by_group[group]) < per_group:
            if not _take_first(candidates[group], selected_by_group[group]):
                break
        if len(selected_by_group[group]) != per_group:
            raise RuntimeError(
                f"Could not select {per_group} parseable documents for {group}"
            )

    return [
        row for group in GROUPS for row in selected_by_group[group][:per_group]
    ]
