# Code/create_project_structure.py
from pathlib import Path
import textwrap

def create_structure(root: Path):
    # Core folders
    folders = [
        "Documents",
        "Data/raw",
        "Data/processed",
        "Code",
        "Figures",
        "References",
    ]

    for f in folders:
        path = root / f
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created {path.relative_to(root)}")

    # README
    readme_path = root / "README.md"
    if not readme_path.exists():
        readme_text = textwrap.dedent("""
        # MBTA Sentiment & Reliability Project

        ## Folder structure

        - `Documents/` — proposal, full report drafts, slides.
        - `Data/raw/` — original exports (Reddit, MBTA alerts, weather).
        - `Data/processed/` — cleaned CSVs ready for analysis.
        - `Code/` — Python scripts and notebooks.
        - `Figures/` — plots and diagrams.
        - `References/` — PDFs of papers and reports.

        See `Data/data_dictionary.md` for variable definitions.
        """).strip() + "\n"
        readme_path.write_text(readme_text)
        print("Created README.md")

    # Data dictionary stub
    data_dict_path = root / "Data" / "data_dictionary.md"
    if not data_dict_path.exists():
        data_dict_text = textwrap.dedent("""
        # Data Dictionary

        This file documents all datasets, variables, units, and basic cleaning/filtering steps.

        ## reddit_mbta_raw.csv
        - `id`: Reddit post ID
        - `author`: Anonymized user name or ID
        - `title`: Post title
        - `body`: Post selftext
        - `score`: Upvotes (net)
        - `created_utc`: Unix timestamp (seconds) in UTC
        - `subreddit`: Subreddit name

        ## mbta_alerts_raw.csv
        - `alert_id`
        - `route`
        - `effect`
        - `severity`
        - `description`
        - `start_time`
        - `end_time`

        ## weather_boston_raw.csv
        - `timestamp`
        - `temp_c`
        - `precip_mm`
        - `condition` (e.g., rain / snow / clear)

        (Add to and refine this as you go!)
        """).strip() + "\n"
        data_dict_path.write_text(data_dict_text)
        print("Created Data/data_dictionary.md")


if __name__ == "__main__":
    # Change this if you want to point directly at your Dropbox path
    project_root = Path(__file__).resolve().parents[1]
    print(f"Project root: {project_root}")
    create_structure(project_root)
