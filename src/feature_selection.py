from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd


def feature_importance(X, y):
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    return importance


def select_top_features(X, importance, top_n=20):
    top_features = importance.head(top_n).index
    return X[top_features]