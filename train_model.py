"""
train_model.py
Trains and evaluates multiple ML models on the Student Career Success dataset.
Target: Placement_Status  (binary classification: Placed / Not Placed)
Saves the best pipeline to models/model.pkl
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ─── Problem definition ───────────────────────────────────────────────────────
TARGET = "Placement_Status"          # binary: Placed / Not Placed
DROP_COLS = [
    "Student_ID",       # identifier
    "Company_Tier",     # leakage (directly derived from Placement_Status)
    "Career_Field",     # leakage
    "Placement_Mode",   # leakage
    "Starting_Salary_USD",  # leakage (0 when not placed)
    "Employability_Score",  # leakage (composite score used to assign placement)
]
PROBLEM_TYPE = "classification"
print(f"  Target  : {TARGET}")
print(f"  Problem : {PROBLEM_TYPE}")

# ─── Feature engineering ──────────────────────────────────────────────────────
df = df.drop(columns=DROP_COLS, errors="ignore")

NUMERIC_FEATURES = [
    "Age",
    "Attendance_Percentage",
    "Study_Hours_Per_Week",
    "CGPA",
    "Programming_Skill",
    "Projects_Completed",
    "Certifications",
    "Hackathons",
    "Internships",
    "Resume_Score",
    "Communication_Skills",
    "Teamwork",
    "Problem_Solving",
    "Interview_Score",
]

ORDINAL_FEATURES = {
    "Academic_Performance": ["Poor", "Average", "Good", "Excellent"],
    "English_Proficiency":  ["Basic", "Intermediate", "Advanced"],
    "University_Year":      ["Freshman", "Sophomore", "Junior", "Senior"],
}

BINARY_FEATURES = ["GitHub_Profile", "Leadership_Experience", "LinkedIn_Profile"]

NOMINAL_FEATURES = ["Gender", "Major"]

# ─── Encode binary Yes/No columns ─────────────────────────────────────────────
for col in BINARY_FEATURES:
    df[col] = (df[col] == "Yes").astype(int)

# One-hot encode nominal categoricals
df = pd.get_dummies(df, columns=NOMINAL_FEATURES, drop_first=False)
nominal_ohe_cols = [c for c in df.columns if any(c.startswith(n+"_") for n in NOMINAL_FEATURES)]

# ─── Build feature list ───────────────────────────────────────────────────────
ordinal_cols   = list(ORDINAL_FEATURES.keys())
ordinal_cats   = list(ORDINAL_FEATURES.values())
all_feature_cols = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + ordinal_cols
    + nominal_ohe_cols
)

X = df[all_feature_cols].copy()
y = df[TARGET].copy()

label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)           # Placed→1, Not Placed→0
classes = label_enc.classes_.tolist()
print(f"  Classes : {classes}")
print(f"  Class distribution: {dict(zip(*np.unique(y_enc, return_counts=True)))}")

# ─── Preprocessor ─────────────────────────────────────────────────────────────
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

ord_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=ordinal_cats, handle_unknown="use_encoded_value", unknown_value=-1)),
])

# Binary + OHE cols are already numeric; just impute + scale
bin_ohe_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num",     num_transformer,     NUMERIC_FEATURES),
    ("ord",     ord_transformer,     ordinal_cols),
    ("bin_ohe", bin_ohe_transformer, BINARY_FEATURES + nominal_ohe_cols),
])

# ─── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

# ─── Model zoo ────────────────────────────────────────────────────────────────
CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Extra Trees":          ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n-- Model Comparison (5-fold CV on training set) --")
for name, clf in CANDIDATES.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
    results[name] = {"cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std()}
    print(f"  {name:<25} CV F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

best_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
print(f"\n  Best model: {best_name}")

# ─── Retrain best model on full train set ─────────────────────────────────────
best_clf = CANDIDATES[best_name]
best_pipe = Pipeline([("pre", preprocessor), ("clf", best_clf)])
best_pipe.fit(X_train, y_train)

# ─── Evaluate on held-out test set ───────────────────────────────────────────
y_pred   = best_pipe.predict(X_test)
y_prob   = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average="weighted")
auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

print(f"\n-- Test-set Metrics --")
print(f"  Accuracy  : {acc:.4f}")
print(f"  F1 (wtd)  : {f1:.4f}")
if auc:
    print(f"  ROC-AUC   : {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=classes))

# ─── Save model + metadata ───────────────────────────────────────────────────
joblib.dump(best_pipe, MODEL_PATH)
print(f"Model saved -> {MODEL_PATH}")

meta = {
    "model_name":       best_name,
    "target":           TARGET,
    "problem_type":     PROBLEM_TYPE,
    "classes":          classes,
    "feature_cols":     all_feature_cols,
    "numeric_features": NUMERIC_FEATURES,
    "binary_features":  BINARY_FEATURES,
    "ordinal_features": ordinal_cols,
    "ordinal_categories": ordinal_cats,
    "nominal_features": NOMINAL_FEATURES,
    "nominal_ohe_cols": nominal_ohe_cols,
    "test_accuracy":    round(acc, 4),
    "test_f1":          round(f1, 4),
    "test_auc":         round(auc, 4) if auc else None,
    "cv_results":       {k: {m: round(v, 4) for m, v in vv.items()} for k, vv in results.items()},
    "label_mapping":    {int(i): c for i, c in enumerate(classes)},
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved -> {META_PATH}")
print("\nTraining complete.")
