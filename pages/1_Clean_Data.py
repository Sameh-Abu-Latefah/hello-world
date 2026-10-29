from pathlib import Path

import pandas as pd
import streamlit as st

from hello_world.cleaning import (
    clean_text,
    drop_missing_rows,
    mark_invalid_ages,
    normalize_gender,
    remove_duplicates,
    split_data,
    to_number,
)
from hello_world.data_utils import (
    duplicate_count,
    load_csv,
    missing_values,
    numeric_text_columns,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "messy_customers.csv"

st.set_page_config(page_title="Clean Data", layout="wide")
st.title("Clean Data")

source = st.radio("Data source", ["Use sample dataset", "Upload a CSV"], horizontal=True)

df = None

if source == "Upload a CSV":
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except ValueError as e:
            st.error(f"Could not read this file: {e}")
else:
    df = load_csv(DATA_FILE)

if df is None:
    st.info("Upload a CSV file or choose the sample dataset to begin.")
    st.stop()

st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")

# ---------- 1. What is wrong with the data ----------
st.header("1. Problems in the data")

st.subheader("Missing values")
missing = missing_values(df)
if missing["missing"].sum() == 0:
    st.success("No missing values.")
else:
    st.dataframe(missing, use_container_width=True)

st.subheader("Duplicate rows")
duplicates = duplicate_count(df)
if duplicates == 0:
    st.success("No duplicate rows.")
else:
    st.warning(f"{duplicates} rows are copies of an earlier row.")
    st.dataframe(df[df.duplicated(keep=False)].sort_values(list(df.columns)),
                 use_container_width=True)

st.subheader("Wrong data types")
wrong_types = numeric_text_columns(df)
if wrong_types:
    st.warning("These columns hold numbers but are stored as text: " + ", ".join(wrong_types))
else:
    st.success("No column has a wrong type.")

# ---------- 2. Clean the rows (before the split) ----------
st.header("2. Clean the rows")
st.caption("These steps follow fixed rules. They learn nothing from the data, so they are safe before the split.")

clean_df = df.copy()

if st.checkbox("Remove duplicate rows", value=True):
    clean_df = remove_duplicates(clean_df)

text_columns = df.select_dtypes(exclude="number").columns.tolist()
tidy = st.multiselect(
    "Text columns to tidy (strip spaces, lowercase)",
    text_columns,
    default=[c for c in ["plan"] if c in text_columns],
)
if tidy:
    clean_df = clean_text(clean_df, tidy)

if "gender" in clean_df.columns and st.checkbox("Unify gender (m / f become male / female)", value=True):
    clean_df = normalize_gender(clean_df)

to_fix = st.multiselect(
    "Columns to convert from text to numbers",
    text_columns,
    default=numeric_text_columns(df),
)
for col in to_fix:
    clean_df[col] = to_number(clean_df[col])

numbers = clean_df.select_dtypes(include="number").columns.tolist()
if "age" in numbers and st.checkbox("Mark impossible ages as missing", value=True):
    low, high = st.columns(2)
    min_age = low.number_input("Youngest allowed age", value=17.0)
    max_age = high.number_input("Oldest allowed age", value=70.0)
    clean_df = mark_invalid_ages(clean_df, "age", min_age, max_age)

st.write(f"{len(df)} rows before, {len(clean_df)} rows after.")
st.dataframe(clean_df.head(10), use_container_width=True)

# ---------- 3. Missing values ----------
st.header("3. Missing values")
st.dataframe(missing_values(clean_df).query("missing > 0"), use_container_width=True)

how = st.radio("What to do with them?", ["Fill them after the split", "Drop the rows"], horizontal=True)

ready_df = clean_df
if how == "Drop the rows":
    with_gaps = clean_df.columns[clean_df.isna().any()].tolist()
    drop_cols = st.multiselect("Drop rows with a missing value in:", with_gaps, default=with_gaps)
    ready_df = drop_missing_rows(clean_df, drop_cols)
    st.write(f"{len(clean_df) - len(ready_df)} rows dropped, {len(ready_df)} left.")
else:
    st.write("They will be filled later, with values learned from the train set only.")

# ---------- 4. Train / test split ----------
st.header("4. Train / test split")

columns = ready_df.columns.tolist()
target = st.selectbox(
    "Target column (the one you want to predict)",
    columns,
    index=columns.index("churned") if "churned" in columns else 0,
)
test_percent = st.slider("Test set size (%)", 10, 40, 20, step=5)

try:
    train, test = split_data(ready_df, target, test_percent / 100)
except ValueError as e:
    st.error(f"Could not split the data: {e}")
    st.stop()

left, right = st.columns(2)
left.metric("Train rows", len(train))
right.metric("Test rows", len(test))

st.write(f"Share of each value of '{target}'. It should be about the same in both sets.")
st.dataframe(
    pd.DataFrame({
        "train": train[target].value_counts(normalize=True).round(3),
        "test": test[target].value_counts(normalize=True).round(3),
    }),
    use_container_width=True,
)