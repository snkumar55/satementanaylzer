from __future__ import annotations

import pandas as pd

def csv_download_bytes(records: pd.DataFrame) -> bytes:
    """Serialize transaction records for a Streamlit CSV download."""
    return records.to_csv(index=False).encode("utf-8")
