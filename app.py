import streamlit as st

from hello_world.data_utils import (
    category_counts,
    column_types,
    histogram_data,
    load_csv,
    missing_values,
    numeric_columns,
)

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")

source = st.radio("Data source", ["Upload a CSV", "Use sample dataset"], horizontal=True)

df = None

if source == "Upload a CSV":
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except ValueError as e:
            st.error(f"Could not read this file: {e}")
else:
    df = load_csv("data/data.csv")

if df is not None:
    st.subheader("First rows")
    st.dataframe(df.head(10), use_container_width=True)

    rows, cols = df.shape
    st.subheader("Shape")
    st.write(f"{rows} rows, {cols} columns")

    st.subheader("Column types")
    st.dataframe(column_types(df), use_container_width=True)
    st.subheader("Summary statistics")

    nums = numeric_columns(df)
    st.sidebar.header("Chart settings")
    column = st.sidebar.selectbox("Column", df.columns)
    chart_type = st.sidebar.selectbox("Chart type", ["Histogram", "Bar (counts)", "Line"])

    st.subheader("Chart")
    series = df[column]
    is_numeric = column in numeric_columns(df)

    if chart_type == "Histogram":
        if is_numeric:
            st.bar_chart(histogram_data(series))
        else:
            st.warning("A histogram needs a numeric column. Pick another column or use Bar (counts).")
    elif chart_type == "Bar (counts)":
        st.bar_chart(category_counts(series))
    else:
        if is_numeric:
            st.line_chart(series.dropna().reset_index(drop=True))
        else:
            st.warning("A line chart needs a numeric column.")
    if nums:
        st.write("Numeric columns")
        st.dataframe(df[nums].describe(), use_container_width=True)
    else:
        st.info("This file has no numeric columns.")

    text_df = df.select_dtypes(exclude="number")
    if len(text_df.columns) > 0:
        st.write("Text columns")
        st.dataframe(text_df.describe(), use_container_width=True)

    st.subheader("Missing values")
    missing = missing_values(df)
    if missing["missing"].sum() == 0:
        st.success("No missing values.")
    else:
        st.dataframe(missing, use_container_width=True)
else:
    st.info("Upload a CSV file or choose the sample dataset to begin.")