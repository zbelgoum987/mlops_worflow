from model.train import read_data
import os
import pytest


def test_csvs_no_files():
    with pytest.raises(RuntimeError) as error:
        read_data("./")
    assert error.match("No CSV files found in provided data")


def test_csvs_no_files_invalid_path():
    with pytest.raises(RuntimeError) as error:
        read_data("/invalid/path/does/not/exist/")
    assert error.match("Cannot use non-existent path provided")


def test_csvs_creates_dataframe():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    datasets_directory = os.path.join(current_directory, 'datasets')
    result = read_data(datasets_directory)
    assert len(result) == 20
