import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

df = pd.read_csv('data-week-3.csv')
df.columns = df.columns.str.lower().str.replace(' ','_')
categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)
categorical_cols += list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_cols:
    df[c] = df[c].str.lower().str.replace(' ','_')
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn=='yes').astype(int)


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

num_f = ['tenure', 'monthlycharges', 'totalcharges']
cat_f = ['gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',]


def train(df_train, y_train, C=1.0):

    X_train_num = df_train[num_f].values
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train_cat = df_train[cat_f].values

    X_train_cat = ohe.fit_transform(X_train_cat)

    X_train = np.column_stack([X_train_num, X_train_cat])

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return scaler, ohe, model

def predict(df, scaler, ohe, model):
    df_num = df[num_f].values
    df_cat = df[cat_f].values
    df_num = scaler.transform(df_num)
    df_cat = ohe.transform(df_cat)

    X = np.column_stack([df_num, df_cat])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    scaler, ohe, model = train(df_train, y_train, C)
    y_pred = predict(df_val, scaler, ohe, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'fold {fold}: AUC = {auc}')
    fold += 1

print('validation results')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print('training final model')

scaler, ohe, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_full_train, scaler, ohe, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

with open(output_file, 'wb') as f_out:
    pickle.dump((scaler, ohe, model), f_out)

print(f'the model is saved to {output_file}')