from utils import patternize

PROFILING_CONFIG = {
    'base': {
        # ---------------------------------------------------------
        # BASIC METRICS
        # ---------------------------------------------------------
        'row_count': lambda s: len(s),
        'property_count': lambda s: s.notna().sum(),
        'absent_property': lambda s: s.isna().sum(),
        # ---------------------------------------------------------
        # VALUE PRESENCE
        # ---------------------------------------------------------
        'value_present': lambda s: s.notna().mean(),
        'value_absent':  lambda s: s.isna().mean(),
        # ---------------------------------------------------------
        # UNIQUE / PATTERN
        # ---------------------------------------------------------
        'unique_pattern_count': lambda s: s.dropna().astype(str).map(patternize).nunique(),
        'unique_values_count': lambda s: s.nunique(dropna=True),
        # ---------------------------------------------------------
        # VALUE COUNTS
        # ---------------------------------------------------------
        'top_value_counts': lambda s: s.dropna().value_counts(ascending=False).iloc[0],
        'bottom_value_counts': lambda s: s.dropna().value_counts(ascending=True).iloc[0],
        'top_pattern_counts': lambda s: s.dropna().astype(str).map(patternize).value_counts(ascending=False).iloc[0],
        'bottom_pattern_counts': lambda s: s.dropna().astype(str).map(patternize).value_counts(ascending=True).iloc[0],
        # ---------------------------------------------------------
        # DISCRETE STATS
        # ---------------------------------------------------------
        'discrete_entropy': lambda s: -sum(
            p * np.log2(p)
            for p in s.dropna().value_counts(normalize=True).values
        ),
        'unique_values': lambda s: s.dropna().nunique() / s.count()
    },
    'numeric': {
        # ---------------------------------------------------------
        # NUMERIC BASIC
        # ---------------------------------------------------------
        'mean': lambda s: s.dropna().astype(float).mean(),
        'sum': lambda s: s.dropna().astype(float).sum(), 
        # ---------------------------------------------------------
        # NUMERIC ADVANCED
        # ---------------------------------------------------------
        'median': lambda s: s.dropna().astype(float).median(),
        'std_dev': lambda s: s.dropna().astype(float).std(ddof=0),
        'mode': lambda s: s.dropna().astype(float).mode().iloc[0],
        'min': lambda s: s.dropna().astype(float).min(),
        'max': lambda s:  s.dropna().astype(float).max(),
        # ---------------------------------------------------------
        # NUMERIC DEEP STATS
        # ---------------------------------------------------------
        'pearson_skewness': lambda s: 3 * (s.dropna().astype(float).mean() - s.dropna().astype(float).median()) / s.dropna().astype(float).std(ddof=0),
        'geometric_mean': lambda s: float(np.exp(np.log(s.dropna().astype(float)).mean())),
        'variance': lambda s: s.dropna().astype(float).var(ddof=0),
    },
    'string': {
        # ---------------------------------------------------------
        # STRING BASIC
        # ---------------------------------------------------------
        'empty_property': lambda s: s.dropna().astype(str).eq('').sum(),
        'average_string_length': lambda s: s.dropna().astype(str).map(len).mean(),
        # ---------------------------------------------------------
        # STRING ADVANCED
        # ---------------------------------------------------------
        'string_length_min': lambda s: s.astype(str).str.len().min(),
        'string_length_max': lambda s: s.astype(str).str.len().max(),
        'lower_case_strings': lambda s: s.dropna().astype(str).map(str.islower).mean(),
        'upper_case_strings': lambda s: s.dropna().astype(str).map(str.isupper).mean(),
        'trivial_values': lambda s: s.dropna().astype(str).str.upper().isin(['NA','N/A','np.nan','NULL']).mean() 
    },
    'datetime': {
        # ---------------------------------------------------------
        # DATETIME DISTRIBUTIONS
        # ---------------------------------------------------------
        'min_timestamp': lambda s: s.min().value // 10**9,
        'max_timestamp': lambda s: s.max().value // 10**9,
    },
    'boolean': {}
}
