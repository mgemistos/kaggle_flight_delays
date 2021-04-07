import warnings
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

PATH_TO_DATA = Path('/home/tsiou/Documents/github/kaggle_flight_delays/data')
train_df = pd.read_csv(PATH_TO_DATA/'flight_delays_train.csv')

test_df = pd.read_csv(PATH_TO_DATA/'flight_delays_test.csv')


train_df['flight'] = train_df['Origin'] +'-->'+train_df['Dest']
#print(train_df.head())

test_df['flight'] = test_df['Origin'] +'-->'+test_df['Dest']
#print(test_df.head())

categorical_feat_idx = np.where(train_df.drop('dep_delayed_15min' , axis=1).dtypes == 'object')[0]
#print(categorical_feat_idx)

X_train = train_df.drop('dep_delayed_15min',axis=1).values
y_train = train_df['dep_delayed_15min'].map({'Y':1,'N':0}).values
X_test = test_df.values

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train,
                                                                test_size=0.3,
                                                                random_state=17)
print(train_df.shape)

ctb = CatBoostClassifier(task_type = 'GPU', random_seed = 17, silent=True)
ctb.fit(X_train_part, y_train_part, cat_features = categorical_feat_idx)
ctb_valid_pred = ctb.predict_proba(X_valid)[:,1]
score_part = roc_auc_score(y_valid, ctb_valid_pred)

ctb.fit(X_train, y_train, cat_features = categorical_feat_idx)
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv',
                             index_col='id')
    sample_sub['dep_delayed_15min'] = ctb_test_pred
    sample_sub.to_csv('ctb_pred.csv')
