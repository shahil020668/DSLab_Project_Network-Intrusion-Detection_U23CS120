# src/models.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


# 🔹 Random Forest
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# 🔹 Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# 🔹 Support Vector Machine
def train_svm(X_train, y_train):
    model = SVC(
        probability=True,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model



# 🔹 Gradient Boosting
def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5
    )
    model.fit(X_train, y_train)
    return model


# 🔥 XGBoost (IMPORTANT MODEL)
def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# 🔥 Train all models together
def get_all_models(X_train, y_train):
    models = {}

    print("Training Random Forest...")
    models['RandomForest'] = train_random_forest(X_train, y_train)

    print("Training Logistic Regression...")
    models['LogisticRegression'] = train_logistic_regression(X_train, y_train)

    print("Training SVM...")
    models['SVM'] = train_svm(X_train, y_train)

    print("Training Gradient Boosting...")
    models['GradientBoosting'] = train_gradient_boosting(X_train, y_train)

    print("Training XGBoost...")
    models['XGBoost'] = train_xgboost(X_train, y_train)

    return models


# 🔹 Safe probability prediction (useful for UI)
def predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    else:
        return None