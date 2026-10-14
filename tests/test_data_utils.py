import io
import pytest
from hello_world.data_utils import load_csv, missing_values


def test_missing_values_counts_blanks():
    df = load_csv(io.StringIO("a,b\n1,\n2,3\n,4\n"))
    result = missing_values(df).set_index("column")
    assert result.loc["a", "missing"] == 1
    assert result.loc["b", "missing"] == 1


def test_load_csv_rejects_empty_file():
    with pytest.raises(ValueError):
        load_csv(io.StringIO(""))