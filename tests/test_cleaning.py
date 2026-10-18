import pandas as pd

import pandas as pd

from hello_world.cleaning import (
    clean_text,
    find_income_outliers,
    fit_income_bounds,
    mark_income_outliers,
    remove_duplicates,
    split_data,
    to_number,
    normalize_gender,
    fit_fill_values,
    fit_encoder,
    apply_encoder
)

def test_remove_duplicates_keeps_one_copy():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert len(remove_duplicates(df)) == 2


def test_clean_text_strips_spaces_and_lowercases():
    df = pd.DataFrame({"plan": [" Basic", "PREMIUM", "standard"]})
    result = clean_text(df, ["plan"])
    assert result["plan"].tolist() == ["basic", "premium", "standard"]


def test_clean_text_keeps_missing_values_missing():
    df = pd.DataFrame({"plan": ["Basic", None]})
    assert clean_text(df, ["plan"])["plan"].isna().tolist() == [False, True]


def test_to_number_removes_commas():
    result = to_number(pd.Series(["1,928", "4924"]))
    assert result.tolist() == [1928.0, 4924.0]


def test_to_number_turns_bad_and_missing_values_into_nan():
    result = to_number(pd.Series(["N/A", "abc", None]))
    assert result.isna().all()


INCOME = pd.DataFrame({"income": [3000.0, 4000.0, 5000.0, 6000.0, 100000.0]})


def test_income_bounds_follow_the_iqr_rule_without_margin():
    # Q1 = 4000, Q3 = 6000, IQR = 2000 -> 4000 - 3000 and 6000 + 3000
    assert fit_income_bounds(INCOME, margin=0) == (1000.0, 9000.0)


def test_income_margin_is_added_to_the_upper_bound_only():
    low, high = fit_income_bounds(INCOME, margin=1000)
    assert (low, high) == (1000.0, 10000.0)


def test_income_lower_bound_is_never_negative():
    df = pd.DataFrame({"income": [10.0, 20.0, 30.0, 40.0, 50.0]})
    assert fit_income_bounds(df, margin=0)[0] == 0


def test_margin_keeps_a_value_just_above_the_iqr_line():
    df = pd.DataFrame({"income": [3000.0, 4000.0, 5000.0, 6000.0, 9500.0]})
    flagged_without = find_income_outliers(df, fit_income_bounds(df, margin=0))
    flagged_with = find_income_outliers(df, fit_income_bounds(df, margin=1000))
    assert flagged_without.iloc[-1] and not flagged_with.iloc[-1]


def test_find_income_outliers_flags_the_big_value_but_not_missing_income():
    df = pd.DataFrame({"income": [3000.0, 4000.0, 5000.0, 6000.0, 100000.0, None]})
    flags = find_income_outliers(df, fit_income_bounds(INCOME))
    assert flags.tolist() == [False, False, False, False, True, False]


def test_mark_income_outliers_sets_nan_and_keeps_the_original_untouched():
    bounds = fit_income_bounds(INCOME)
    result = mark_income_outliers(INCOME, bounds)
    assert result["income"].isna().tolist() == [False] * 4 + [True]
    assert INCOME["income"].iloc[-1] == 100000.0


def test_income_bounds_learned_on_train_are_applied_to_test():
    train = pd.DataFrame({"income": [3000.0, 4000.0, 5000.0, 6000.0, 7000.0]})
    test = pd.DataFrame({"income": [4500.0, 50000.0]})
    bounds = fit_income_bounds(train)
    assert find_income_outliers(test, bounds).tolist() == [False, True]



SPLIT_DF = pd.DataFrame({"id": range(100), "churned": [1] * 20 + [0] * 80})


def test_split_sizes_follow_test_size():
    train, test = split_data(SPLIT_DF, test_size=0.3)
    assert len(train) == 70 and len(test) == 30


def test_split_keeps_the_same_churn_share_in_both_parts():
    train, test = split_data(SPLIT_DF, test_size=0.2)
    assert train["churned"].mean() == 0.2
    assert test["churned"].mean() == 0.2


def test_split_has_no_overlap_and_loses_no_rows():
    train, test = split_data(SPLIT_DF)
    ids = train["id"].tolist() + test["id"].tolist()
    assert sorted(ids) == list(range(100))


def test_split_gives_the_same_result_with_the_same_seed():
    first = split_data(SPLIT_DF, seed=7)[1]["id"].tolist()
    second = split_data(SPLIT_DF, seed=7)[1]["id"].tolist()
    assert first == second

def test_normalize_gender_unifies_case_spaces_and_short_forms():
        df = pd.DataFrame({"gender": ["Male", "MALE", " male ", "M", "f", "FEMALE", "Female "]})
        result = normalize_gender(df)
        assert result["gender"].tolist() == ["male"] * 4 + ["female"] * 3

def test_normalize_gender_keeps_missing_values_missing():
        df = pd.DataFrame({"gender": ["M", None]})
        result = normalize_gender(df)
        assert result["gender"].iloc[0] == "male"
        assert pd.isna(result["gender"].iloc[1])

def test_normalize_gender_does_not_touch_other_values():
        df = pd.DataFrame({"gender": ["Other"]})
        assert normalize_gender(df)["gender"].tolist() == ["other"]

def test_normalize_gender_does_not_change_the_original():
        df = pd.DataFrame({"gender": ["M"]})
        normalize_gender(df)
        assert df["gender"].iloc[0] == "M"

def test_fit_fill_values_median_and_constant():
    train = pd.DataFrame({
        "age": [20.0, 30.0, 100.0, None],
        "city": ["a", None, "b", None],
    })
    values = fit_fill_values(train, {"age": "median", "city": "Unknown"})
    assert values == {"age": 30.0, "city": "Unknown"}

def test_fit_encoder_learns_the_categories_from_train():
    train = pd.DataFrame({"plan": ["premium", "basic", "basic", None]})
    assert fit_encoder(train, ["plan"]) == {"plan": ["basic", "premium"]}


def test_apply_encoder_makes_one_column_per_category_and_drops_the_original():
    df = pd.DataFrame({"plan": ["basic", "premium", "basic"], "age": [20, 30, 40]})
    result = apply_encoder(df, {"plan": ["basic", "premium"]})
    assert list(result.columns) == ["age", "plan_basic", "plan_premium"]
    assert result["plan_basic"].tolist() == [1, 0, 1]
    assert result["plan_premium"].tolist() == [0, 1, 0]


def test_apply_encoder_gives_test_the_same_columns_as_train():
    train = pd.DataFrame({"plan": ["basic", "premium"]})
    test = pd.DataFrame({"plan": ["gold"]})
    categories = fit_encoder(train, ["plan"])
    train_cols = list(apply_encoder(train, categories).columns)
    test_encoded = apply_encoder(test, categories)
    assert list(test_encoded.columns) == train_cols
    assert test_encoded.iloc[0].sum() == 0


def test_apply_encoder_does_not_change_the_original():
    df = pd.DataFrame({"plan": ["basic"]})
    apply_encoder(df, {"plan": ["basic"]})
    assert list(df.columns) == ["plan"]