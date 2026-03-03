from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Handles loading and cleaning of market data.
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    def load_csv(self, file_path: Path | str) -> pd.DataFrame:
        """
        Load market data from a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Cleaned pandas DataFrame with datetime index.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If required columns are missing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(path)

        # Basic schema validation
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            # Check if index is already set maybe? Assuming flat CSV for now as per req
            raise ValueError(f"Missing required columns: {missing_cols}")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "Dataset must contain 'timestamp' column or have a datetime index"
            )

        # Sort and remove duplicates
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        return df
