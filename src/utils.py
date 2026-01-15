import re
import dataiku
import joblib
import io

def patternize(s: str) -> str:
    if s is None or s != s:
        return ''
    s = str(s)
    protected_chars = {
        '@': '\x00AT\x00',
        '.': '\x00DOT\x00',
        ':': '\x00COLON\x00',
        '/': '\x00SLASH\x00',
        '-': '\x00DASH\x00',
        '_': '\x00UNDERSCORE\x00'
    }
    for ch, token in protected_chars.items():
        s = s.replace(ch, token)

    s = re.sub(r'[a-z]', 'a', s)    
    s = re.sub(r'[A-Z]', 'A', s)    
    s = re.sub(r'\d', '9', s)       
    s = re.sub(r'[;!?]', 'P', s)    
    s = re.sub(r'[^aA9P\x00]', '_', s)

    for ch, token in protected_chars.items():
        s = s.replace(token, ch)
    return s

DATETIME_FORMATS = {
    '%Y-%m-%d': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
    '%d.%m.%Y': re.compile(r'^\d{2}\.\d{2}\.\d{4}$'),
    '%Y/%m/%d': re.compile(r'^\d{4}/\d{2}/\d{2}$'),
    '%Y-%m-%d %H:%M:%S': re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'),
    '%Y-%m-%d %H:%M:%S.%f': re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$'),
    '%Y-%m-%d %H:%M:%S.%f%z': re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:?\d{2}$'),
    '%Y-%m-%dT%H:%M:%S.%fZ': re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$'),
}

def get_dtype_key(dtype: str):
    if dtype in ['int64', 'float64']:
        return 'numeric'
    if dtype == 'object':
        return 'string'
    if str(dtype).startswith('datetime'):
        return 'datetime'
    if dtype == 'bool':
        return 'boolean'
    return dtype

dq_rules = [
    'min_in_range',
    'max_in_range',
    'avg_in_range',
    'sum_in_range',
    'median_in_range',
    'std_dev_in_range',
    'values_are_not_empty',
    'values_are_empty',
    'values_are_unique',
    'values_in_set',
    'top_N_values_in_set',
    'most_frequent_value_in_set',
    'matches_pattern'
]

def load_model(file_name: str):
    folder_path = 'model'
    folder = dataiku.Folder(folder_path)
    
    with folder.get_download_stream(file_name) as stream:
        model_bytes = io.BytesIO(stream.read())
    model_bytes.seek(0)
    return joblib.load(model_bytes)
