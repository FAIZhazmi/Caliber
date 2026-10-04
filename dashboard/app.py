"""Open Dwi's dashboard from the existing Streamlit entry point."""

import runpy
from pathlib import Path

# Both launchers use the unified dashboard, including the Supabase REST data path.

runpy.run_path(
    str(Path(__file__).resolve().with_name("executive_app.py")),
    run_name="__main__",
)
