# Tools

Standalone diagnostic utilities that support the pipeline but are not part
of the numbered sequence.

| File | Purpose |
|---|---|
| `check_db7_labels_against_raw_data.py` | Settles NinaPro DB7's true movement count empirically against your actual downloaded `.mat` files (bypassing the windowing/feature pipeline entirely), rather than trusting documentation or code defaults. This is what motivated the DB7 movement-count fix recorded in `dev_history/`. Run with `python tools/check_db7_labels_against_raw_data.py` from the project root. |
| `db7_label_diagnostic.txt` | The saved report from the script above. |
