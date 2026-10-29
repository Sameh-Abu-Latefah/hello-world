import pandas as pd
from sklearn.model_selection import train_test_split


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exact copies of an earlier row."""
    return df.drop_duplicates().reset_index(drop=True)


def clean_text(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Strip spaces and lowercase the given text columns. Missing values stay missing."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].str.strip().str.lower()
    return out

def to_number(series: pd.Series) -> pd.Series:
    """'1,928' becomes 1928.0. Anything that is not a number becomes NaN."""
    text = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")

def fit_income_bounds(train: pd.DataFrame, column: str = "income",
                      k: float = 1.5, margin: float = 1000) -> tuple:
    """IQR bounds for income, learned from the data you pass (use train only).

    lower = Q1 - k*IQR (never below 0)
    upper = Q3 + k*IQR + margin, so real high incomes just above the IQR line stay.
    """
    q1, q3 = train[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    return max(q1 - k * iqr, 0), q3 + k * iqr + margin


def find_income_outliers(df: pd.DataFrame, bounds: tuple, column: str = "income") -> pd.Series:
    """True for each row whose income is outside the bounds. Missing income is not an outlier."""
    low, high = bounds
    return (df[column] < low) | (df[column] > high)


def mark_income_outliers(df: pd.DataFrame, bounds: tuple, column: str = "income") -> pd.DataFrame:
    """Set income outside the bounds to NaN. Fill it later, after the split."""
    out = df.copy()
    out[column] = out[column].mask(find_income_outliers(df, bounds, column))
    return out

def mark_invalid_ages(df: pd.DataFrame, column: str = "age",
                      min_age: float = 17, max_age: float = 70) -> pd.DataFrame:
    """Set ages outside [min_age, max_age] to NaN. Fill them later, after the split."""
    out = df.copy()
    out[column] = out[column].where(out[column].between(min_age, max_age))
    return out

def split_data(df: pd.DataFrame, target: str = "churned",
               test_size: float = 0.2, seed: int = 42) -> tuple:
    """Split into (train, test). Both keep the same share of each target value."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)

def fit_scaler(train: pd.DataFrame, columns: list) -> dict:
    """Mean and standard deviation of each column, learned from train only."""
    return {col: (train[col].mean(), train[col].std(ddof=0)) for col in columns}


def apply_scaler(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """(value - train mean) / train standard deviation. Works on train and on test."""
    out = df.copy()
    for col, (mean, std) in params.items():
        out[col] = (out[col] - mean) / std if std else out[col] - mean
    return out


GENDER_MAP = {"m": "male", "f": "female"}


def normalize_gender(df: pd.DataFrame, column: str = "gender") -> pd.DataFrame:
    """Strip spaces, lowercase, and spell out 'm' and 'f'. Missing values stay missing."""
    out = clean_text(df, [column])
    out[column] = out[column].replace(GENDER_MAP)
    return out

def fit_fill_values(train: pd.DataFrame, strategies: dict) -> dict:
    """Decide what to put in the empty cells of each column, using train only.

    strategies: {"age": "median", "city": "Unknown"}
    "median" is computed from train. Anything else is used as is.
    """
    return {
        col: train[col].median() if how == "median" else how
        for col, how in strategies.items()
    }
def apply_fill_values(df: pd.DataFrame, values: dict) -> pd.DataFrame:
    """Fill the empty cells with the values learned on train. Works on train and on test."""
    return df.fillna(values)

def fit_encoder(train: pd.DataFrame, columns: list) -> dict:
    """The categories each column has in train. Learned from train only."""
    return {col: sorted(train[col].dropna().unique()) for col in columns}


def apply_encoder(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """One 0/1 column per known category. A category never seen in train gets all zeros."""
    out = df.copy()
    for col, known in categories.items():
        for category in known:
            out[f"{col}_{category}"] = (out[col] == category).astype(int)
        out = out.drop(columns=col)
    return out
def drop_missing_rows(df: pd.DataFrame, columns: list | None = None) -> pd.DataFrame:
    """Drop rows that have an empty cell in the given columns (all columns if none given)."""
    return df.dropna(subset=columns).reset_index(drop=True)

def prepare_data(df: pd.DataFrame, target: str = "churned",
                 test_size: float = 0.2, seed: int = 42) -> tuple:
    """Run every cleaning step in the safe order and return (train, test).

    1. Fixed rules on all rows: duplicates, text, numbers, impossible ages.
    2. Split.
    3. Learn from train only (outlier bounds, fill values, scaler, encoder), apply to both.
    """
    df = remove_duplicates(df)
    if "plan" in df.columns:
        df = clean_text(df, ["plan"])
    if "gender" in df.columns:
        df = normalize_gender(df)
    if "income" in df.columns:
        df["income"] = to_number(df["income"])
    if "age" in df.columns:
        df = mark_invalid_ages(df)

    train, test = split_data(df, target, test_size, seed)

    if "income" in train.columns:
        bounds = fit_income_bounds(train)
        train = mark_income_outliers(train, bounds)
        test = mark_income_outliers(test, bounds)

    features = [c for c in train.columns if c != target]
    numbers = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    texts = [c for c in features if c not in numbers]

    strategies = {c: "median" for c in numbers}
    strategies.update({c: "Unknown" for c in texts})
    fill_values = fit_fill_values(train, strategies)
    train = apply_fill_values(train, fill_values)
    test = apply_fill_values(test, fill_values)

    scaler = fit_scaler(train, numbers)
    train = apply_scaler(train, scaler)
    test = apply_scaler(test, scaler)

    encoder = fit_encoder(train, texts)
    return apply_encoder(train, encoder), apply_encoder(test, encoder)



