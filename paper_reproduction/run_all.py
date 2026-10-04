"""Regenerate every table and figure of the article from the shipped result files (no NinaPro data needed)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ("p13_article_tables_and_figures.py", "make_figure5_and_table_s13.py"):
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
print("\nAll tables and figures regenerated in outputs/tables and outputs/figures.")
