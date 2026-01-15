import pandas as pd
import numpy as np
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel

df_profiles = pd.concat(
    [
        df_num_profile,
        df_str_profile, 
        df_dt_profile, 
        df_bool_profile
    ],
    ignore_index=True)

df_clustered_profile = (
    df_profiles
    .merge(df_kmeans_all, on=keys, how='left')
    .merge(df_dbscan_all, on=keys, how='left')
)
df_clustered_profile

POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

def make_kmeans_lf(rule, min_conf, max_conf=None):
    name = f'lf_kmeans_{rule}_{min_conf}_{max_conf}'
    @labeling_function(name=name)
    def lf(x):
        cluster_type, cluster_id = x.kmeans_cluster.split('_')
        if (x.kmeans_confidence >= min_conf) and (max_conf is None or x.kmeans_confidence < max_conf):
            if rule in kmeans_mapping_dicts[cluster_type].get(int(cluster_id), []):
                return POSITIVE
        return ABSTAIN
    return lf

def make_dbscan_lf(rule, min_conf, max_conf=None):
    name = f'lf_dbscan_{rule}_{min_conf}_{max_conf}'
    @labeling_function(name=name)
    def lf(x):
        cluster_type, cluster_id = x.dbscan_cluster.split('_')
        if (x.dbscan_confidence >= min_conf) and (max_conf is None or x.dbscan_confidence < max_conf):
            if rule in dbscan_mapping_dicts[cluster_type].get(int(cluster_id), []):
                return POSITIVE
        return ABSTAIN
    return lf

kmeans_mapping_dicts = {
    'numeric': kmeans_label_numeric,
    'string': kmeans_label_string,
    'datetime': kmeans_label_datetime,
    'boolean': kmeans_label_boolean
}
dbscan_mapping_dicts = {
    'numeric': dbscan_label_numeric,
    'string': dbscan_label_string,
    'datetime': dbscan_label_datetime,
    'boolean': dbscan_label_boolean
}

n_epochs = 500
random_state = 42

all_binary_labels = []
all_lfs = []
L_all = []
for rule in dq_rules:
    lfs = [
        make_kmeans_lf(rule, 0.8),               
        make_kmeans_lf(rule, 0.5, 0.8),              
        make_dbscan_lf(rule, 0.8),                
        make_dbscan_lf(rule, 0.5, 0.8),               
    ]
    all_lfs.extend(lfs)
    applier = PandasLFApplier(lfs)
    L = applier.apply(df_clustered_profile)
    L_all.append((rule, lfs, L))

    label_model = LabelModel(cardinality=2)
    label_model.fit(L, n_epochs=n_epochs, seed=random_state)

    binary_labels = label_model.predict(L)
    all_binary_labels.append(binary_labels)
