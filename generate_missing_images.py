"""
Generate missing visualization images for the presentation
- Feature Importance (Top 20)
- Correlation Heatmap
- ROC Curve Comparison (All Models Overlaid)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc
import joblib
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("📊 Generating missing presentation images...\n")

# ============================================================
# 1. FEATURE IMPORTANCE (TOP 20)
# ============================================================
print("1️⃣ Generating Feature Importance Plot...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Train Extra Trees for feature importance
et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)

# Get feature importance
importance = pd.Series(et.feature_importances_, index=X_train.columns)
top_20 = importance.nlargest(20)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
top_20.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/eda/feature_importance_top20.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/eda/feature_importance_top20.png\n")

# ============================================================
# 2. CORRELATION HEATMAP
# ============================================================
print("2️⃣ Generating Correlation Heatmap...")

# Select top 15 features for clarity in heatmap
top_15_features = importance.nlargest(15).index.tolist()
X_train_top = X_train[top_15_features]

# Calculate correlation matrix
corr_matrix = X_train_top.corr()

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Top 15 Features', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/eda/correlation_heatmap.png\n")

# ============================================================
# 3. ROC CURVE COMPARISON (ALL MODELS)
# ============================================================
print("3️⃣ Generating ROC Curve Comparison...")

models_list = ['RandomForest', 'LogisticRegression', 'SVM', 'GradientBoosting', 'XGBoost']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

fig, ax = plt.subplots(figsize=(10, 8))

# Plot diagonal line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)

# Load and plot ROC curve for each model
for model_name, color in zip(models_list, colors):
    model = joblib.load(f'models/{model_name}.pkl')
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})', color=color)
    else:
        print(f"   ⚠️  {model_name} doesn't support predict_proba")

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
plt.tight_layout()
plt.savefig('outputs/roc_curves/roc_curve_comparison_all.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/roc_curves/roc_curve_comparison_all.png\n")

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("✨ ALL IMAGES GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\n📍 IMAGE LOCATIONS FOR YOUR PPT:\n")
print("🔵 Page 5 (EDA Section):")
print("   1. Class Distribution:        outputs/eda/attack_distribution.png ✅")
print("   2. Feature Importance (Top 20): outputs/eda/feature_importance_top20.png ✅")
print("   3. Correlation Heatmap:        outputs/eda/correlation_heatmap.png ✅")
print("\n🔵 Page 7 (Results Section):")
print("   4. Model Comparison Chart:     outputs/model_comparison/model_comparison.png ✅")
print("   5. XGBoost Confusion Matrix:   outputs/confusion_matrix/XGBoost_confusion_matrix.png ✅")
print("   6. ROC Curve Comparison:       outputs/roc_curves/roc_curve_comparison_all.png ✅")
print("\nℹ️  All images are 300 DPI (high quality for printing)")
print("=" * 60)
