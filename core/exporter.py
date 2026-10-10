"""Export service for synthetic datasets.

Supports in-memory exports to CSV, Excel (.xlsx via openpyxl), and JSON,
preserving data types, headers, and handling missing values cleanly.
"""

from __future__ import annotations
import io
import json
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd


class ExportService:
    """Handles dataset serialization and in-memory file export."""

    @staticmethod
    def generate_filename(domain: str, task: str, extension: str) -> str:
        """Create a clean, standardized filename with domain, task, and timestamp."""
        clean_domain = domain.lower().replace(" ", "_").replace("&", "and")
        clean_task = task.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"synthetic_{clean_domain}_{clean_task}_{timestamp}.{extension}"

    @classmethod
    def to_csv(
        cls, df: pd.DataFrame, domain: str = "dataset", task: str = "data"
    ) -> Tuple[bytes, str, str]:
        """Export DataFrame to CSV in memory.

        Returns:
            Tuple of (file_bytes, mime_type, filename)
        """
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        csv_bytes = buffer.getvalue().encode("utf-8")
        filename = cls.generate_filename(domain, task, "csv")
        return csv_bytes, "text/csv", filename

    @classmethod
    def to_excel(
        cls,
        df: pd.DataFrame,
        domain: str = "dataset",
        task: str = "data",
        sheet_name: str = "SyntheticData",
    ) -> Tuple[bytes, str, str]:
        """Export DataFrame to Excel (.xlsx) in memory using openpyxl.

        Returns:
            Tuple of (file_bytes, mime_type, filename)
        """
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        excel_bytes = buffer.getvalue()
        filename = cls.generate_filename(domain, task, "xlsx")
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return excel_bytes, mime, filename

    @classmethod
    def to_json(
        cls,
        df: pd.DataFrame,
        domain: str = "dataset",
        task: str = "data",
        orient: str = "records",
        indent: int = 2,
    ) -> Tuple[bytes, str, str]:
        """Export DataFrame to JSON in memory.

        Returns:
            Tuple of (file_bytes, mime_type, filename)
        """
        # Convert Timestamp or NaNs for clean JSON serialization
        json_str = df.to_json(orient=orient, date_format="iso", indent=indent)
        json_bytes = json_str.encode("utf-8")
        filename = cls.generate_filename(domain, task, "json")
        return json_bytes, "application/json", filename
