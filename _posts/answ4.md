```
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np

# you can access datasets by calling:
# data_train = pd.read_csv("data/data_train.csv")
# data_test = pd.read_csv("data/data_test.csv")

# binarize: yes/no -> 1/0
def binarize(df):
    df_ = df.copy()
    binarize_cols = [col for col in df_.columns if df_[col].dropna().astype("str").str.lower().isin(["yes", "no"]).all()]
    for col in binarize_cols:
        df_[col] = df_[col].astype("str").str.lower().map({"yes":1, "no":0}).astype("int")
    return df_

def identify_customers(data_train, data_test):
    # (1) Binarize columns
    data_train_bin = binarize(data_train)
    data_test_bin = binarize(data_test)

    # (2) (X/y) x (Train/Test)
    X_train = data_train_bin.drop("label", axis=1)
    X_cols = X_train.columns
    y_train = data_train_bin["label"].astype("int")
    if "label" in data_test_bin.columns:
        X_test = data_test_bin.drop("label", axis=1)
    else:
        X_test = data_test_bin.copy()
    X_test = X_test[X_cols]

    # (3) One-hot encoded train/test
    onehot_train = X_train.copy()
    onehot_test = X_test.copy()

    # (4) Label proportion (1 vs.0)
    prop = round(y_train.mean(), 3)

    # (5-1) Logistic regression (lr)
    lr = LogisticRegression(class_weight = {0: prop, 1: 1-prop},
    random_state =0, max_iter = 50)
    lr.fit(X_train, y_train)
    coef = lr.coef_.ravel()
    negative_impact = [col for col, w in zip(X_cols, coef) if w < 0]

    # (5-2) Random forest (rf)
    rf = RandomForestClassifier(max_depth = 10, random_state = 0,
    n_estimators = 30, class_weight = {0: prop, 1: 1-prop})
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_K = 5
    top_idx = np.argsort(importances)[::-1][:top_K]
    feature_importance = [(X_cols[i], float(importances[i])) for i in top_idx]

    # (6) Recall (train & test)
    lr_recall_train = float(recall_score(y_train, lr.predict(X_train)))
    rf_recall_train = float(recall_score(y_train, rf.predict(X_train)))
    lr_recall_test = float(recall_score(data_test["label"].astype("int") if "label" in data_test.columns else pd.Series(np.zeros(len(X_test), dtype="int"), index=X_test.index), lr.predict(X_test)) if "label" in data_test.columns else 0)
    rf_recall_test = float(recall_score(data_test["label"].astype("int") if "label" in data_test.columns else pd.Series(np.zeros(len(X_test), dtype="int"), index=X_test.index), rf.predict(X_test)) if "label" in data_test.columns else 0)

    lr_recall = (lr_recall_train, lr_recall_test)
    rf_recall = (rf_recall_train, rf_recall_test)

    # (7) Predicted probabilities
    lr_probs = lr.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]
    lr_probs = pd.Series(lr_probs, index = X_test.index)
    rf_probs = pd.Series(rf_probs, index = X_test.index)
    lr_obs = lr_probs.sort_values(ascending = False).index
    rf_obs = rf_probs.sort_values(ascending = False).index

    return {
        "onehot_train": onehot_train,
        "onehot_test": onehot_test,
        "prop": prop,
        "negative_impact": negative_impact,
        "feature_importance": feature_importance,
        "lr_recall": lr_recall,
        "rf_recall": rf_recall,
        "lr_obs": lr_obs,
        "rf_obs": rf_obs,
    }

```

