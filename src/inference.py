import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple 
from dataclasses import dataclass 
from dataiku.core.sql import SQLExecutor2
from utils import dq_rules, load_model

opt_classifier = load_model('opt_model.pkl')

@dataclass
class RuleSuggestion:
    column_name: str
    rule_type: str
    confidence: float
    parameter_rho: Any
    violation_condition_sql: str
    validation_sql: str
    sample_sql: str

class InferencePipeline:
    def __init__(self, connection_str: str, prob_threshold: float=0.5, error_tolerance: float=0.0):
        self.connection_str = connection_str
        self.threshold = prob_threshold
        self.error_tolerance = error_tolerance
        
        self.executor = SQLExecutor2(connection=self.connection_str)
        self.profiler_config = PROFILING_CONFIG
        self.rule_names = dq_rules
        self.classifier = opt_classifier

        self.RULE_FUNCS = {
            'avg_in_range': self._check_avg_in_range,
            'sum_in_range': self._check_sum_in_range,
            'median_in_range': self._check_median_in_range,
            'std_dev_in_range': self._check_std_dev_in_range,
            'min_in_range': self._check_min_in_range,
            'max_in_range': self._check_max_in_range,
            'values_in_set': self._check_values_in_set,
            'top_N_values_in_set': self._check_top_N_values_in_set,
            'most_frequent_value_in_set': self._check_most_frequent_value_in_set,
            'matches_pattern': self._check_matches_pattern,
            'values_are_not_empty': self._check_values_are_not_empty,
            'values_are_empty': self._check_values_are_empty,
            'values_are_unique': self._check_values_are_unique
        }

    def run(self, table_name: str, limit: int = 100000) -> List[RuleSuggestion]:
        query = f'SELECT * FROM {table_name} LIMIT {limit}'
        df = self.executor.query_to_df(query)
        if df.empty:
            print('[WARN] No data loaded.')
            return []

        profiler = DataProfiler(df, self.profiler_config, table_name)
        profiles = profiler.run()
        features = profiler.extract_features()

        probs_list = self.classifier.predict_proba(features)

        suggestions = []
        for col_idx, profile in enumerate(profiles):
            col_name = profile.metadata.column
            for label_idx, rule_name in enumerate(self.rule_names):
                prob = probs_list[label_idx][col_idx][1]
                if prob <= self.threshold:
                    continue
                suggestion = self._create_suggestion(
                    col_name=col_name,
                    s=df[col_name],
                    rule_type=rule_name,
                    confidence=prob,
                    table_name=table_name
                )
                if suggestion:
                    suggestions.append(suggestion)
        return suggestions

    def _create_suggestion(self, col_name: str, s: pd.Series, rule_type: str, confidence: float, table_name: str) -> RuleSuggestion:
        rho = None
        violation_condition = ''
        aggregate_sql_func = ''
        try:
            s = s.dropna()
            result = self.RULE_FUNCS[rule_type](s=s, col=col_name)
            rho = result['rho']
            if 'aggregate_sql_func' in result.keys():
                aggregate_sql_func = result['aggregate_sql_func']
            else:
                violation_condition = result['violation_condition']    
        except Exception as e:
            print(f"[ERROR] Error for column {col_name} and rule {rule_type}: {e}")
            return None
         
        validation_sql = ''
        sample_sql = ''
        if aggregate_sql_func:
            lower_bound = float(rho) * 0.9
            upper_bound = float(rho) * 1.1
            validation_sql = (
                f"SELECT CASE "
                f"WHEN {aggregate_sql_func} NOT BETWEEN {lower_bound} AND {upper_bound} THEN 1.0 "
                f"ELSE 0.0 END as error_rate "
                f"FROM {table_name}"
            )
            sample_sql = (
                f"SELECT '{rule_type}' as check_type, "
                f"{rho} as expected_value, "
                f"{aggregate_sql_func} as actual_value "
                f"FROM {table_name}"
            )
        else:
            if rule_type == 'values_are_unique':
                validation_sql = (
                    f"SELECT (COUNT({col_name}) - COUNT(DISTINCT {col_name})) / CAST(COUNT({col_name}) AS FLOAT) as error_rate "
                    f"FROM {table_name}"
                )
                sample_sql = (
                    f"SELECT {col_name}, COUNT(*) as cnt FROM {table_name} "
                    f"GROUP BY {col_name} HAVING COUNT(*) > 1 LIMIT 10"
                )
            else:
                validation_sql = (
                    f"SELECT CAST(SUM(CASE WHEN {violation_condition} THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as error_rate "
                    f"FROM {table_name}"
                )
                sample_sql = (
                    f"SELECT * FROM {table_name} "
                    f"WHERE {violation_condition} "
                    f"LIMIT 10"
                )
        return RuleSuggestion(
            column_name=col_name,
            rule_type=rule_type,
            confidence=confidence,
            parameter_rho=rho,
            violation_condition_sql=violation_condition if not aggregate_sql_func else f"Aggregate check vs {rho}",
            validation_sql=validation_sql,
            sample_sql=sample_sql
        )

    def _format_sql_list(self, values: List[Any]) -> str:
        formatted = [f"'{str(v)}'" if isinstance(v, str) else str(v) for v in values]
        return f"({', '.join(formatted)})"
    
    def _check_avg_in_range(self, s: pd.Series, col: str):
        if pd.api.types.is_numeric_dtype(s):
            rho = s.mean()
            sql_func = f"AVG({col})"
        elif pd.api.types.is_datetime64_any_dtype(s):
            raise TypeError("AVG für datetime nicht sinnvoll")
        else:
            rho = s.astype(str).map(len).mean()
            sql_func = f"AVG(LENGTH({col}))"
        return { "rho": rho, "aggregate_sql_func": sql_func }

    def _check_sum_in_range(self, s: pd.Series, col: str):
        if pd.api.types.is_numeric_dtype(s):
            rho = s.sum()
            sql_func = f"SUM({col})"
        elif pd.api.types.is_datetime64_any_dtype(s):
            raise TypeError("SUM für datetime nicht sinnvoll")
        else:
            rho = s.astype(str).map(len).sum()
            sql_func = f"SUM(LENGTH({col}))"
        return { "rho": rho, "aggregate_sql_func": sql_func }

    def _check_median_in_range(self, s: pd.Series, col: str):
        if pd.api.types.is_numeric_dtype(s):
            rho = s.median()
            sql_func = f"APPROX_MEDIAN({col})"
        elif pd.api.types.is_datetime64_any_dtype(s):
            raise TypeError("Median für datetime nicht sinnvoll")
        else:
            rho = s.astype(str).map(len).median()
            sql_func = f"APPROX_MEDIAN(LENGTH({col}))"
        return { "rho": rho, "aggregate_sql_func": sql_func }

    def _check_std_dev_in_range(self, s: pd.Series, col: str):
        if pd.api.types.is_numeric_dtype(s):
            rho = s.std()
            sql_func = f"STDDEV({col})"
        elif pd.api.types.is_datetime64_any_dtype(s):
            raise TypeError("STDDEV für datetime nicht sinnvoll")
        else:
            rho = s.astype(str).map(len).std()
            sql_func = f"STDDEV(LENGTH({col}))"
        return { "rho": rho, "aggregate_sql_func": sql_func }

    def _check_min_in_range(self, s: pd.Series, col: str):
        if pd.api.types.is_numeric_dtype(s):
            rho = s.min()
            violation_condition = f"{col} < {rho}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            rho = s.min().strftime('%Y-%m-%d %H:%M:%S')
            violation_condition = f"{col} < TIMESTAMP('{rho}')"
        else:
            rho = s.astype(str).map(len).min()
            violation_condition = f"LENGTH({col}) < {rho}"
        return {"rho": rho, "violation_condition": violation_condition}

    def _check_max_in_range(self, s: pd.Series, col: str):
        if pd.api.types.is_numeric_dtype(s):
            rho = s.max()
            violation_condition = f"{col} > {rho}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            rho = s.max().strftime('%Y-%m-%d %H:%M:%S')
            violation_condition = f"{col} > TIMESTAMP('{rho}')"
        else:
            rho = s.astype(str).map(len).max()
            violation_condition = f"LENGTH({col}) > {rho}"
        return {"rho": rho, "violation_condition": violation_condition}

    def _check_values_in_set(self, s: pd.Series, col: str):
        rho = s.unique().tolist()
        sql_set = self._format_sql_list(rho)
        violation_condition = f"{col} NOT IN {sql_set}"
        return {"rho": rho, "violation_condition": violation_condition}
    
    def _check_top_N_values_in_set(self, s: pd.Series, col: str):
        rho = s.value_counts().head(5).index.tolist()
        sql_set = self._format_sql_list(rho)
        violation_condition = f"{col} NOT IN {sql_set}"
        return {"rho": rho, "violation_condition": violation_condition}
    
    def _check_most_frequent_value_in_set(self, s: pd.Series, col: str):
        value_counts = s.value_counts()
        rho = [value_counts.idxmax()]
        sql_set = self._format_sql_list(rho)
        violation_condition = f"{col} NOT IN {sql_set}"
        return {"rho": rho, "violation_condition": violation_condition}

    def _check_matches_pattern(self, s: pd.Series, col: str):
        pattern_counts = s.astype(str).map(patternize).value_counts()
        rho = pattern_counts.idxmax() if not pattern_counts.empty else None
        violation_condition = f"{col} NOT REGEXP '{rho}'" if rho else None
        return {"rho": rho, "violation_condition": violation_condition}

    def _check_values_are_not_empty(self, s: pd.Series, col: str):
        rho = 1.0
        violation_condition = f"{col} IS NULL"
        return {"rho": rho, "violation_condition": violation_condition}

    def _check_values_are_empty(self, s: pd.Series, col: str):
        rho = 0.0
        violation_condition = f"{col} IS NOT NULL"
        return {"rho": rho, "violation_condition": violation_condition}

    def _check_values_are_unique(self, s: pd.Series, col: str):
        rho = 1.0
        violation_condition = "DUPLICATE_CHECK"
        return {"rho": rho, "violation_condition": violation_condition}
