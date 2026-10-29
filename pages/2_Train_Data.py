from pathlib import Path

import streamlit as st

from hello_world.cleaning import prepare_data
from hello_world.data_utils import load_csv
from hello_world.training import (
    baseline_accuracy,
    evaluate_model,
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "messy_customers.csv"

st.set_page_config(page_title="Train Model", layout="wide")
st.title("Train Model")

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

# ---------- 1. Clean and split ----------
st.header("1. Clean and split the data")

TARGET = "churned"
if TARGET not in df.columns:
    st.error(f"This file has no '{TARGET}' column.")
    st.stop()
target = TARGET
st.write(f"Target: **{target}**")

test_percent = st.slider("Test set size (%)", 10, 40, 20, step=5)

try:
    train, test = prepare_data(df, target, test_percent / 100)
except (ValueError, KeyError) as e:
    st.error(f"Could not prepare the data: {e}")
    st.stop()

left, right = st.columns(2)
left.metric("Train rows", len(train))
right.metric("Test rows", len(test))


# ---------- 2. Choose features and model ----------
st.header("2. Choose features and model")

candidates = [c for c in train.columns if c not in (target, "customer_id")]
features = st.multiselect(
    "Features (what the model learns from)", candidates, default=candidates
)
if not features:
    st.warning("Pick at least one feature.")
    st.stop()

model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

if model_name == "Logistic Regression":
    model = train_logistic_regression(X_train, y_train)
elif model_name == "Decision Tree":
    depth = st.slider("Max depth", 1, 15, 3)
    model = train_decision_tree(X_train, y_train, depth)
else:
    depth = st.slider("Max depth", 1, 15, 3)
    trees = st.slider("Number of trees", 10, 300, 100, step=10)
    model = train_random_forest(X_train, y_train, trees, depth)

# ---------- 3. Results ----------
st.header("3. Results")

result = evaluate_model(model, X_train, y_train, X_test, y_test)
baseline = baseline_accuracy(y_train, y_test)

col1, col2, col3 = st.columns(3)
col1.metric("Train accuracy", f"{result['train_accuracy']:.1%}")
col2.metric("Test accuracy", f"{result['test_accuracy']:.1%}")
col3.metric("Baseline (always the most common class)", f"{baseline:.1%}")