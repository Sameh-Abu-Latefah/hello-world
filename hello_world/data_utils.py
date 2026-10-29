import numpy as np
import pandas as pd


def load_csv(file) -> pd.DataFrame:
    """Read a CSV and return a DataFrame. Raise ValueError with a clear message if it fails."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The file is not a valid CSV.")
    except UnicodeDecodeError:
        raise ValueError("The file could not be read as text. Is it really a CSV?")

    if df.empty or len(df.columns) == 0:
        raise ValueError("The file has no data rows.")
    return df


def column_types(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column with its type."""
    return pd.DataFrame({
        "column": df.columns,
        "type": [str(t) for t in df.dtypes],
    })


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percent for each column."""
    counts = df.isna().sum()
    return pd.DataFrame({
        "column": counts.index,
        "missing": counts.values,
        "percent": (counts.values / len(df) * 100).round(1),
    })


def numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include="number").columns.tolist()

def histogram_data(series: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Count how many values fall in each range. Blanks are ignored."""
    counts, edges = np.histogram(series.dropna(), bins=bins)
    labels = [f"{edges[i]:.1f} to {edges[i + 1]:.1f}" for i in range(len(counts))]
    return pd.DataFrame({"count": counts}, index=labels)


def category_counts(series: pd.Series, top: int = 20) -> pd.DataFrame:
    """Count each distinct value. Keeps the 20 most common."""
    return series.value_counts().head(top).to_frame("count")

def duplicate_count(df: pd.DataFrame) -> int:
    """How many rows are exact copies of an earlier row."""
    return int(df.duplicated().sum())


def numeric_text_columns(df: pd.DataFrame, min_share: float = 0.8) -> list:
    """Text columns whose values are mostly numbers, like '1,928'. These have the wrong type."""
    found = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        text = df[col].dropna().astype(str).str.replace(",", "", regex=False)
        if len(text) and pd.to_numeric(text, errors="coerce").notna().mean() >= min_share:
            found.append(col)
    return found