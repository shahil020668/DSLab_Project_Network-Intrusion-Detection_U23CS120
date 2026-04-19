# src/main.py

import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import get_all_models
from src.evaluation import evaluate_model


# 🔹 Load processed data
def load_data():
    print("📂 Loading processed data...")

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    print("✅ Data loaded successfully")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# 🔹 Save trained models
def save_models(models):
    print("\n💾 Saving models...")

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        file_path = f"models/{name}.pkl"
        joblib.dump(model, file_path)
        print(f"✔ Saved {name} → {file_path}")

    print("✅ All models saved successfully")


# 🔹 Ensure output folders exist
def create_output_dirs():
    os.makedirs("outputs/confusion_matrix", exist_ok=True)
    os.makedirs("outputs/roc_curves", exist_ok=True)
    os.makedirs("outputs/model_comparison", exist_ok=True)


# 🔹 Save model comparison chart
def save_model_comparison(results):
    comparison_df = pd.DataFrame(results).set_index("model_name")

    plt.figure(figsize=(10, 6))
    comparison_df[["accuracy", "precision", "recall", "f1"]].plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/model_comparison/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# 🔹 Main pipeline
def main():
    print("🚀 Starting Intrusion Detection System\n")

    # Step 0: Create folders for images
    create_output_dirs()

    # Step 1: Load Data
    X_train, X_test, y_train, y_test = load_data()

    # Step 2: Train Models
    print("\n🧠 Training all models...")
    models = get_all_models(X_train, y_train)

    # Step 3: Evaluate Models (this will SAVE images)
    print("\n📊 Evaluating all models...")
    evaluation_results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        evaluation_results.append(metrics)

    # Step 3.5: Save model comparison image
    save_model_comparison(evaluation_results)

    # Step 4: Save Models
    save_models(models)

    print("\n🎉 Pipeline completed successfully!")
    print("📁 Check 'outputs/' folder for saved graphs")


# 🔹 Run script
if __name__ == "__main__":
    main()