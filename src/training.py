import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss
from lightgbm import LGBMClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import optuna

random_state = 42

TOTAL_CORES = os.cpu_count()
TRIALS_PARALLEL = int(n_trials * 0.1)
CORES_PER_TRIAL = int(TOTAL_CORES / TRIALS_PARALLEL)
n_trials, TOTAL_CORES, TRIALS_PARALLEL, CORES_PER_TRIAL

# grouping
tables_unique = df_clustered_profile['table_name'].unique()

train_tables, test_tables = train_test_split(
    tables_unique,
    test_size=0.2,
    random_state=random_state
)

train_mask = df_clustered_profile['table_name'].isin(train_tables)
test_mask = df_clustered_profile['table_name'].isin(test_tables)

X_train, y_train = X[train_mask], Y[train_mask]
X_test, y_test = X[test_mask], Y[test_mask]

y_train = np.where(y_train == 1, 1, 0)
y_test = np.where(y_test == 1, 1, 0)

# base
baseline_model = MultiOutputClassifier(
    LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        verbose=-1,
        random_state=random_state,
        n_jobs=TOTAL_CORES
    )
)

baseline_model.fit(X_train, y_train)

# hpo
k = 5
n_trials = 1000

SEARCH_SPACE = {
    'num_leaves': ('int', (10, 60)),
    'max_depth': ('int', (5, 15)),
    'min_child_samples': ('int', (20, 100)),
    'lambda_l2': ('float', (0.0, 10.0)),
    'learning_rate': ('float', (0.01, 0.2)),
    'n_estimators': ('int', (50, 500)),
    'colsample_bytree': ('float', (0.5, 1.0)),
}

df_clustered_profile_split = df_clustered_profile[train_mask].copy()
df_clustered_profile_split['table_group'] = df_clustered_profile_split['table_name']

Y_table_train = pd.DataFrame(y_train, index=df_clustered_profile_split.index)
Y_group_train = Y_table_train.groupby(df_clustered_profile_split['table_group']).max()

groups_train = Y_group_train.index.values
group_labels_train = Y_group_train.values

def get_params(search_space, trial):
    params = {}
    for name, (ptype, bounds) in search_space.items():
        if ptype == 'int':
            params[name] = trial.suggest_int(name, bounds[0], bounds[1])
        elif ptype == 'float':
            params[name] = trial.suggest_float(name, bounds[0], bounds[1])
    return params

    
def objective(trial):
    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    params = get_params(SEARCH_SPACE, trial)
    losses = []
    for fold_idx, (train_idx, valid_idx) in enumerate(mskf.split(groups_train, group_labels_train)):
        train_mask_fold = df_clustered_profile_split['table_group'].isin(groups_train[train_idx])
        valid_mask_fold = df_clustered_profile_split['table_group'].isin(groups_train[valid_idx])

        X_train, y_train_hard, w_train = X[train_mask], Y_hard[train_mask], Y_probs[train_mask]
        X_test, y_test_hard, w_test = X[test_mask], Y_hard[test_mask], Y_probs[test_mask]

        X_train_fold, y_train_fold, w_train_fold = X_train[train_mask_fold], y_train[train_mask_fold], w_train[train_mask_fold]
        X_valid_fold, y_valid_fold, w_valid_fold = X_train[valid_mask_fold], y_train[valid_mask_fold], w_train[valid_mask_fold]

        model = MultiOutputClassifier(
            lgb.LGBMClassifier(
                **params,
                objective='binary',
                boosting_type='gbdt',
                verbose=-1,
                random_state=random_state,
                n_jobs=CORES_PER_TRIAL
            )
        )
        model.fit(X_train_fold, y_train_fold, sample_weight=w_train.mean(axis=1))

        losses.append(hamming_loss(y_valid_fold, model.predict(X_valid_fold)))
    return np.mean(losses)

study = optuna.create_study(
    direction='minimize', 
    sampler=optuna.samplers.TPESampler(seed=random_state)
)
study.optimize(
    objective, 
    n_trials=n_trials, 
    n_jobs=TRIALS_PARALLEL
)

# opt
opt_model = MultiOutputClassifier(
    lgb.LGBMClassifier(
        **study.best_params,
        objective='binary',
        boosting_type='gbdt',
        verbose=-1,
        random_state=random_state,
        n_jobs=TOTAL_CORES
    )
)
opt_model.fit(X_train, y_train)
