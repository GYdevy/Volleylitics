import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# ======================================
# LOAD DATA
# ======================================
CSV_PATH = r"D:\Volleyballey\WhistleDetector\ambiguous_dataset.csv"

df = pd.read_csv(CSV_PATH)
print("Loaded samples:", len(df))

# Features: everything except filename + label
feature_columns = [c for c in df.columns if c not in ["filename", "label"]]
X = df[feature_columns]
y = df["label"].astype(int)

print("\nUsing features:")
print(feature_columns)


# ======================================
# TRAIN / TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# ======================================
# CANDIDATE MODELS + GRID SEARCH
# ======================================
models = {
    "LogisticRegression": GridSearchCV(
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000))
        ]),
        param_grid={
            'clf__C': [0.01, 0.1, 1, 5, 10]
        },
        cv=5
    ),

    "RandomForest": GridSearchCV(
        RandomForestClassifier(),
        param_grid={
            'n_estimators': [200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        },
        cv=5
    ),

    "GradientBoosting": GridSearchCV(
        GradientBoostingClassifier(),
        param_grid={
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [2, 3]
        },
        cv=5
    ),

    "SVM_RBF": GridSearchCV(
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True))
        ]),
        param_grid={
            'clf__C': [0.5, 1, 10],
            'clf__gamma': ['scale', 0.1, 0.01]
        },
        cv=5
    ),

    "KNN": GridSearchCV(
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier())
        ]),
        param_grid={
            'clf__n_neighbors': [3, 5, 7, 9],
            'clf__weights': ['uniform', 'distance']
        },
        cv=5
    ),

    "DecisionTree": GridSearchCV(
        DecisionTreeClassifier(),
        param_grid={
            'max_depth': [None, 6, 10],
            'min_samples_split': [2, 5, 10]
        },
        cv=5
    ),

    "NaiveBayes": GaussianNB()
}


# ======================================
# TRAIN + EVALUATE ALL MODELS
# ======================================
best_model = None
best_auc = -1

results = {}

for name, model in models.items():
    print(f"\n==============================")
    print(f"Training model: {name}")
    print(f"==============================")

    # Fit model
    model.fit(X_train, y_train)

    # Predict & evaluate
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    auc = roc_auc_score(y_test, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("AUC:", auc)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    results[name] = auc

    if auc > best_auc:
        best_auc = auc
        best_model = model


# ======================================
# FINAL BEST MODEL
# ======================================
print("\n====================================")
print("✔ BEST MODEL FOUND:", type(best_model).__name__)
print("✔ BEST AUC:", best_auc)
print("====================================")


# ======================================
# SAVE MODEL + SCALER
# ======================================
MODEL_OUT = r"D:\Volleyballey\WhistleDetector\ambiguous_best_model.pkl"
SCALER_OUT = r"D:\Volleyballey\WhistleDetector\ambiguous_best_scaler.pkl"

if isinstance(best_model, Pipeline):
    scaler = best_model.named_steps["scaler"]
    clf = best_model.named_steps["clf"]
else:
    scaler = None
    clf = best_model

joblib.dump(clf, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)

print("Saved:", MODEL_OUT)
print("Saved:", SCALER_OUT)


# ======================================
# FEATURE IMPORTANCES
# ======================================
print("\n=== FEATURE IMPORTANCES ===")

if hasattr(clf, "feature_importances_"):
    for name, val in zip(feature_columns, clf.feature_importances_):
        print(f"{name:15s}: {val:.4f}")

elif hasattr(clf, "coef_"):
    for name, val in zip(feature_columns, clf.coef_[0]):
        print(f"{name:15s}: {val:.4f}")

else:
    print("Model has no feature importance attribute.")
