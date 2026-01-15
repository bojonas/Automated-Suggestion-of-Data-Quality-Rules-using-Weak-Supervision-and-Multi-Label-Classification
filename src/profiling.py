import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
from metrics import PROFILING_CONFIG
from dataclasses import dataclass
from utils import patternize, get_dtype_key, DATETIME_FORMATS

@dataclass
class Metadata:
    column: str
    table_name: str
    dtype: str
    semantic_dtype: str
        
@dataclass       
class Feature:
    name: str
    value: Union[int, float]

@dataclass
class ColumnProfile:
    metadata: Metadata
    features: Dict[str, Union[int, float]]
        
class DataProfiler:
    def __init__(self, df: pd.DataFrame, metrics_config: dict, table_name: str):
        self.df = df
        self.metrics_config = metrics_config
        self.table_name = table_name
        self.profiles = []

    def run(self) -> List[ColumnProfile]:
        profiles = []
        for col in self.df.columns:
            s = self.df[col]
            dtype, semantic_dtype, s = self.infer_semantic_dtype(s)
            if dtype != semantic_dtype:
                print(f'[INFO] Changed datatype of column {col} from {dtype} to {semantic_dtype}')
                
            metadata = Metadata(
                column=col,
                table_name=self.table_name,
                dtype=dtype,
                semantic_dtype=semantic_dtype
            )

            features = {}
            for category, metrics in self.metrics_config.items():
                for metric_name, metric_func in metrics.items():
                    if category not in ['base', semantic_dtype]:
                        features[metric_name] = np.nan
                        continue
                        
                    try:
                        features[metric_name] = metric_func(s)
                    except Exception as e:
                        print(f'[WARN] Error while calculating metric {metric_name} for column {col}: {e}')
                        features[metric_name] = np.nan

            profile = ColumnProfile(
                features=features,
                metadata=metadata
            )
            profiles.append(profile)
        
        self.profiles = profiles
        return profiles

    def infer_semantic_dtype(self, s: pd.Series):
        dtype = get_dtype_key(s.dtype)
        s = s.dropna()

        normalized = s.astype(str).str.strip().str.lower()
        boolean_values = {
            'true': True,
            't': True,
            '1': True,
            'false': False,
            'f': False,
            '0': False
        }
        if normalized.isin(boolean_values.keys()).all():
            return dtype, 'boolean', normalized.map(boolean_values)

        if dtype != 'string':
            return dtype, dtype, s

        try:
            return dtype, 'numeric', pd.to_numeric(s)
        except (ValueError, TypeError) as e:
            print(f'[WARN] Error while convertig dtype {dtype} to numeric: {e}')
            pass

        try:
            for fmt, pattern in DATETIME_FORMATS.items():
                if s.dropna().astype(str).str.match(pattern).all():
                    return dtype, 'datetime', pd.to_datetime(s, format=fmt, errors='coerce')
        except (ValueError, TypeError) as e:
            print(f'[WARN] Error while convertig dtype {dtype} to datetime with format {fmt}: {e}')
            pass

        return dtype, dtype, s

    def extract_features(self) -> pd.DataFrame:
        return pd.DataFrame([profile.features for profile in self.profiles])
    
    def profiles_to_dataframe(self) -> pd.DataFrame:
        rows = []
        for profile in self.profiles:
            metadata_dict = vars(profile.metadata)
            feature_dict = profile.features
            row = {**metadata_dict, **feature_dict}
            rows.append(row)
        return pd.DataFrame(rows)
