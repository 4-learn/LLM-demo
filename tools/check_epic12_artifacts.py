#!/usr/bin/env python3
"""Check EPIC #12 course inventory, diagrams, demo anchors and session pairs."""
from __future__ import annotations

import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAPPING = ROOT / "docs/course-mapping.csv"
LONG = {"14", "16", "23", "27", "28", "33", "35", "38", "39", "40", "42", "43", "44", "45", "46", "48", "E01", "E02", "E03", "E07", "E09"}
EXPECTED_SESSIONS = {lesson_id: (3 if lesson_id == "42" else 2) for lesson_id in LONG}


def main() -> int:
    with MAPPING.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    errors: list[str] = []
    lessons: dict[str, pathlib.Path] = {}
    mapping_by_id = {row["new_id"]: row for row in rows}
    legacy_no_diagram = {"04", "08", "11", "13", "20", "25", "41", "E04", "E05", "E06", "07", "15", "17", "18", "19", "21", "22", "24", "26", "29", "30", "31", "32", "34", "36", "37", "47", "49", "E08", "50"}
    if len(rows) != 60 or len({row["new_id"] for row in rows}) != 60:
        errors.append(f"mapping must contain 60 unique lessons; found {len(rows)} rows")

    core_ids = {f"{number:02d}" for number in range(1, 52)}
    elective_ids = {f"E{number:02d}" for number in range(1, 10)}
    mapped_ids = {row["new_id"] for row in rows}
    if mapped_ids != core_ids | elective_ids:
        errors.append("mapping IDs do not equal the expected core 01-51 plus elective E01-E09 set")

    def lesson_id_from_path(path: pathlib.Path) -> str | None:
        match = re.match(r"^(\d{2}|E\d{2})-", path.name)
        if match:
            return match.group(1)
        return "51" if path.name == "51-acceptance.md" else None

    lesson_paths = {lesson_id_from_path(path): path for path in ROOT.glob("*.md") if lesson_id_from_path(path)}
    for row in rows:
        lesson_id = row["new_id"]
        lesson_path = lesson_paths.get(lesson_id)
        if not lesson_path or not lesson_path.is_file():
            errors.append(f"{lesson_id}: missing lesson markdown")
            continue
        lessons[lesson_id] = lesson_path
        text = lesson_path.read_text(encoding="utf-8")
        blocks = re.findall(r"```mermaid\n(.*?)\n```", text, re.S)
        if not blocks and lesson_id not in legacy_no_diagram and mapping_by_id[lesson_id]["status"] != "existing":
            errors.append(f"{lesson_id}: no Mermaid diagrams")
        for index, block in enumerate(blocks, 1):
            if not re.search(r"^\s*(?:flowchart|graph)\s+", block, re.M):
                errors.append(f"{lesson_id}: Mermaid block {index} lacks flowchart/graph declaration")
        if text.count("```mermaid") != len(blocks):
            errors.append(f"{lesson_id}: unclosed Mermaid fence")
        anchors = re.findall(r"<!--\s*demo:\s*([^\s]+\.py)\s*-->", text)
        if not anchors:
            errors.append(f"{lesson_id}: missing demo anchor")
        for name in anchors:
            if not (ROOT / name).is_file():
                errors.append(f"{lesson_id}: missing demo file {name}")

    for lesson_id in LONG:
        path = lessons.get(lesson_id)
        if not path:
            continue
        text = path.read_text(encoding="utf-8")
        sessions = list(re.finditer(r"^### (?:Call )?[Ss]ession\s+\d+[^\n]*", text, re.M))
        if len(sessions) != EXPECTED_SESSIONS[lesson_id]:
            errors.append(f"{lesson_id}: found {len(sessions)} sessions; expected {EXPECTED_SESSIONS[lesson_id]}")
        for index, match in enumerate(sessions):
            end = sessions[index + 1].start() if index + 1 < len(sessions) else len(text)
            section = text[match.start():end]
            charts = re.findall(r"```mermaid\n(.*?)\n```", section, re.S)
            if len(charts) != 2:
                errors.append(f"{lesson_id}: {match.group()} has {len(charts)} diagrams, expected 2")
            if not any(label in section for label in ("系統層", "System layer")):
                errors.append(f"{lesson_id}: {match.group()} missing system-layer chart label")
            if not any(label in section for label in ("函式內部", "Internal function")):
                errors.append(f"{lesson_id}: {match.group()} missing internal-function chart label")
    print(f"checked {len(lessons)}/{len(rows)} mapped lessons; {len(LONG)} long-demo lessons")
    if errors:
        print("\n".join(errors))
        return 1
    print("all lesson flowcharts, demo anchors, and long-session pairs passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
