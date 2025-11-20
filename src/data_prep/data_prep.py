# preparing input data: reading from file, normalization, splitting into training and testing sets

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np
from data_prep.input_data_models import DataPrepConfig
from data_prep.constants import *
from config.paths import *
from data_prep.input_data_models import Point

class DataPrep:
    def __init__(self, config: DataPrepConfig):
        self.is_data_prepared: bool = False
        self.config: DataPrepConfig = config

    def get_normalized_input_data_split(self) -> Tuple[List[Point], List[Point]]:
        if(not self.is_data_prepared):
            self._prep_data()
        return self._training_data, self._testing_data
    
    def get_normalized_input_data(self) -> List[Point]:
        if(not self.is_data_prepared):
            self._prep_data()
        return self.normalized_data

    def _prep_data(self):
        self._read_data()
        self._normalize_data()
        self._fill_in_points() # each point will have real and normalized values
        self._split_data()
        self.is_data_prepared = True
    
    def _read_data(self) -> List[List[float]]:
        data_frame = pd.read_csv(PATH_RAW_INPUT_DIR + PATH_RAW_INPUT_FILE)
        self.raw_data = data_frame.values.tolist()
    
    def _normalize_data(self) -> None:
        if(self.config.norm_method == NormMethod.ZSCORE):
            self._normalize_zscore()
        elif(self.config.norm_method == NormMethod.MINMAX):
            self._normalize_minmax()
        else:
            raise ValueError(f"Unknown normalization method: {self.config.norm_method}")
    
    def _normalize_zscore(self) -> None:
        raw_np = np.array(self.raw_data)
        mean = np.mean(raw_np, axis=0)
        std = np.std(raw_np, axis=0)
        zscore_mean = self.config.norm_factors[0]
        zscore_std = self.config.norm_factors[1]
        normalized_np = ((raw_np - mean) / std) * zscore_std + zscore_mean
        self.normalized_data = normalized_np.tolist()
        self.normalization_factors = mean, std
        
    def _normalize_minmax(self) -> None:
        data_min = np.min(self.raw_data)
        data_max = np.max(self.raw_data)
        minmax_min = self.config.norm_factors[0]
        minmax_max = self.config.norm_factors[1]
        self.normalized_data = ((self.raw_data - data_min) / (data_max - data_min)) * (minmax_max - minmax_min) + minmax_min
        self.normalization_factors = data_min, data_max

    def _fill_in_points(self) -> None:
        self.points: List[Point] = []
        for real, norm in zip(self.raw_data, self.normalized_data):
            features_real = real[:-1]
            target_real = real[-1]
            features_norm = norm[:-1]
            target_norm = norm[-1]
            point = Point(
                features_real_values=features_real,
                target_real_value=target_real,
                features_norm_values=features_norm,
                target_norm_value=target_norm
            )
            self.points.append(point)

    def _split_data(self) -> None:
        np.random.seed(self.config.random_seed)
        np.random.shuffle(self.points)
        split_index = int(len(self.points) * (1 - self.config.split_test_size))
        self._training_data = self.points[:split_index]
        self._testing_data = self.points[split_index:]