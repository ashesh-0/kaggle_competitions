import numpy as np
import pandas as pd
from data_preprocessing import DataProcessor


def test_peak_data():
    data = pd.Series([1, 20, 2, -20, 3, 1, -30, 0, 0, 9, 1, 3])
    maximas = [1, 9]
    minimas = [3, 6]
    peak_threshold = 8
    data = DataProcessor.peak_data(data, peak_threshold)
    assert len(data['indices']) == len(maximas) + len(minimas)
    assert set(maximas + minimas) == set(data['indices'])
    assert all(np.array(data['maxima_minima']) == np.array([1, -1, -1, 1]))


def test_corona_identification():
    data = pd.Series([1, 14, 2, -15, 3, 1, -26, 1, 24, 1, 3])
    peak_threshold = 8
    corona_max_distance = 5
    corona_max_height_ratio = 0.4
    pairs = DataProcessor.corona_discharge_index_pairs(
        data,
        peak_threshold,
        corona_max_distance,
        corona_max_height_ratio,
    )
    assert len(pairs) == 2
    assert pairs[0][0] == 1
    assert pairs[0][1] == 3
    assert pairs[1][0] == 6
    assert pairs[1][1] == 8


def test_remove_corona_discharge():
    ser = pd.Series([1, 14, 2, -15, 3, 1, 4, 5, -26, 1, 5, 30, 1])
    ser.index += 100
    peak_threshold = 8
    corona_max_distance = 3
    corona_max_height_ratio = 0.5
    corona_cleanup_distance = 2

    cleaned_ser = DataProcessor.remove_corona_discharge(
        ser,
        peak_threshold,
        corona_max_distance,
        corona_max_height_ratio,
        corona_cleanup_distance,
    )
    assert cleaned_ser.iloc[0] == ser.iloc[0]
    assert all(cleaned_ser.iloc[3 + corona_cleanup_distance:] == ser.iloc[3 + corona_cleanup_distance:])
    assert all(cleaned_ser.iloc[1:3 + corona_cleanup_distance:] == 1)
