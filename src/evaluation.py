# src/evaluation.py

import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_output_dirs(base_dir="outputs"):
    os.makedirs(os.path.join(base_dir, "confusion_matrix"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "roc_curves"), exist_ok=True)


# 🔹 Print full classification report
def print_metrics(y_true, y_pred):
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred))


# 🔹 Plot confusion matrix
def plot_confusion(y_true, y_pred, model_name="Model", save_dir="outputs"):
    cm = confusion_matrix(y_true, y_pred)

    _ensure_output_dirs(save_dir)
    file_path = os.path.join(save_dir, "confusion_matrix", f"{model_name}_confusion_matrix.png")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# 🔹 Calculate AUC score safely
def calculate_auc(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, probs)
    else:
        print("AUC not available for this model")
        return None


# 🔹 Plot ROC Curve
def plot_roc_curve(model, X_test, y_test, model_name="Model", save_dir="outputs"):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)

        _ensure_output_dirs(save_dir)
        file_path = os.path.join(save_dir, "roc_curves", f"{model_name}_roc_curve.png")

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=model_name)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        print("ROC curve not available for this model")


# 🔥 MASTER FUNCTION (USE THIS IN main.py)
def evaluate_model(model, X_test, y_test, model_name="Model", save_dir="outputs"):
    print(f"\n{'='*40}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*40}")

    y_pred = model.predict(X_test)

    # Basic Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Detailed report
    print_metrics(y_test, y_pred)

    # Confusion Matrix
    plot_confusion(y_test, y_pred, model_name, save_dir=save_dir)

    # AUC Score
    auc = calculate_auc(model, X_test, y_test)
    if auc:
        print(f"AUC Score: {auc:.4f}")

    # ROC Curve
    plot_roc_curve(model, X_test, y_test, model_name, save_dir=save_dir)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }