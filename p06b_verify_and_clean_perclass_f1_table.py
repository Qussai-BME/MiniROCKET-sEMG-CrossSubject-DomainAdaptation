"""
p06b_verify_and_clean_perclass_f1_table.py
===================================
Checks outputs/tables/TableS6_perclass_f1_db7.csv for the phantom "class 41"
row that the old (buggy) n_classes=41 would have produced (all-zero row at
the end, corresponding to a movement that never existed in the raw data).

This does NOT require re-running any LOSO training — it just inspects and,
if needed, cleans the already-computed CSV file. Cheap and instant.

USAGE:
    python p06b_verify_and_clean_perclass_f1_table.py

OUTPUT:
  Prints a report. If a phantom row is found, writes a cleaned copy to
  outputs/tables/TableS6_perclass_f1_db7_CLEANED.csv (does not overwrite
  the original, so nothing is lost if something looks off).
"""
import os
import csv
import sys

CSV_PATH = os.path.join("outputs", "tables", "TableS6_perclass_f1_db7.csv")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[INFO] {CSV_PATH} not found. Trying alternate common locations...")
        candidates = [
            os.path.join("tables", "TableS6_perclass_f1_db7.csv"),
            "TableS6_perclass_f1_db7.csv",
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            print("[FAIL] Could not find TableS6_perclass_f1_db7.csv anywhere expected.")
            print("       Tell me the correct path and I'll adjust this script.")
            sys.exit(1)
        csv_path = found
    else:
        csv_path = CSV_PATH

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    print(f"[INFO] Loaded {csv_path}")
    print(f"[INFO] Total rows: {len(rows)}")
    print(f"[INFO] Columns: {fieldnames}")
    print()

    if len(rows) == 40:
        print("[RESULT] Exactly 40 rows — no phantom row present. Nothing to clean.")
        print("         This table is already correct as-is.")
        return

    if len(rows) != 41:
        print(f"[RESULT] Unexpected row count ({len(rows)}, expected 40 or 41).")
        print("         Do not auto-clean this — send me the full file content instead.")
        return

    # 41 rows: check whether the LAST row is the all-zero phantom class
    last_row = rows[-1]
    print("[INFO] Last row (potential phantom):", last_row)

    numeric_cols = [c for c in fieldnames if c.lower() not in ("class_id", "class_name")]
    is_all_zero = all(
        float(last_row[c]) == 0.0 for c in numeric_cols if last_row.get(c, "") not in ("", None)
    )

    if not is_all_zero:
        print("[RESULT] 41 rows found, but the last row is NOT all-zero.")
        print("         This does not match the expected phantom-class pattern —")
        print("         do not auto-clean. Send me this file's full content instead.")
        return

    print("[RESULT] Confirmed: 41 rows, last row is the all-zero phantom class.")
    print("         Writing a cleaned version with that row removed...")

    cleaned_rows = rows[:-1]
    out_path = csv_path.replace(".csv", "_CLEANED.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"[OK] Cleaned file written to: {out_path}")
    print(f"     ({len(cleaned_rows)} rows, phantom row removed)")
    print()
    print("Once you've confirmed this looks right, replace the original file with")
    print("this cleaned version (or tell me and I'll fold it into the paper directly).")


if __name__ == "__main__":
    main()
