import pandas as pd
import pytest

from volbit.data.loader import DataLoader


@pytest.fixture
def sample_csv_path(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("""timestamp,open,high,low,close,volume
2023-01-01 00:00:00,16500.0,16550.0,16480.0,16520.0,100.5
2023-01-01 01:00:00,16520.0,16580.0,16510.0,16550.0,150.2
2023-01-01 02:00:00,16550.0,16600.0,16540.0,16590.0,200.0
""")
    return p


def test_loader_initialization():
    """Test that DataLoader can be initialized."""
    loader = DataLoader()
    assert loader is not None


def test_load_csv_success(sample_csv_path):
    """Test loading a valid CSV file."""
    loader = DataLoader()
    df = loader.load_csv(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "close" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df.index)


def test_load_csv_missing_file():
    """Test loading a missing file raises FileNotFoundError."""
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_csv("non_existent.csv")


def test_data_validation_schema(sample_csv_path):
    """Test that loaded data has correct schema."""
    loader = DataLoader()
    df = loader.load_csv(sample_csv_path)
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        assert col in df.columns


def test_duplicate_handling(tmp_path):
    """Test handling of duplicate timestamps."""
    csv_content = """timestamp,open,high,low,close,volume
2023-01-01 00:00:00,100,110,90,105,10
2023-01-01 00:00:00,100,110,90,105,10
2023-01-01 01:00:00,105,115,100,110,20
"""
    p = tmp_path / "dup.csv"
    p.write_text(csv_content)

    loader = DataLoader()
    df = loader.load_csv(p)
    assert len(df) == 2  # Duplicates removed
    assert df.index.is_unique
