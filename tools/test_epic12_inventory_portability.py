from __future__ import annotations
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_epic12_inventory", ROOT / "tools/build_epic12_inventory.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class InventoryMappingTests(unittest.TestCase):
    def test_default_mapping_is_repository_local_and_complete(self):
        self.assertEqual(BUILDER.DEFAULT_MAPPING, ROOT / "docs/course-mapping.csv")
        with BUILDER.DEFAULT_MAPPING.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({row["new_id"] for row in rows}), 60)
        self.assertEqual(sum(row["track"] == "core" for row in rows), 51)
        self.assertEqual(sum(row["track"] == "elective" for row in rows), 9)

    def test_missing_explicit_mapping_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "absent.csv"
            with self.assertRaisesRegex(SystemExit, "pass --mapping PATH"):
                BUILDER.main(["--mapping", str(missing), "--check"])


if __name__ == "__main__":
    unittest.main()
