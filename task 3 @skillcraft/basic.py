# ============================================================
# TASK 03 - BANK MARKETING DECISION TREE CLASSIFIER
# SkillCraft Technology Internship
# ============================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD YOUR DOWNLOADED DATASET
# ============================================================

# Make sure bank-additional-full.csv is in the same folder
# as this Python file.

DATA_FILE = "bank-additional-full.csv"

df = pd.read_csv(DATA_FILE, sep=";")

print("=" * 60)
print("BANK MARKETING DATASET")
print("=" * 60)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Replace unknown values with NaN
df = df.replace("unknown", np.nan)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# Fill categorical missing values using mode
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


print("\nDataset Shape After Cleaning:")
print(df.shape)


# ============================================================
# 4. TARGET VARIABLE
# ============================================================

# Convert target variable:
# yes = 1
# no  = 0

df["y"] = df["y"].map({
    "yes": 1,
    "no": 0
})

print("\nTarget Distribution:")
print(df["y"].value_counts())

print("\nTarget Percentage:")
print(
    (df["y"].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

# -------------------------------
# Target Distribution
# -------------------------------

plt.figure(figsize=(7, 5))

counts = df["y"].value_counts()

plt.bar(
    ["No Subscription", "Subscription"],
    counts.reindex([0, 1])
)

plt.title("Bank Term Deposit Subscription")
plt.xlabel("Subscription Status")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# -------------------------------
# Age Distribution
# -------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["age"],
    bins=30,
    edgecolor="black"
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# -------------------------------
# Subscription by Job
# -------------------------------

job_rate = (
    df.groupby("job")["y"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))

plt.bar(
    job_rate.index,
    job_rate.values * 100
)

plt.title("Subscription Rate by Job")
plt.xlabel("Job")
plt.ylabel("Subscription Rate (%)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()
plt.show()


# -------------------------------
# Subscription by Education
# -------------------------------

education_rate = (
    df.groupby("education")["y"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

plt.bar(
    education_rate.index,
    education_rate.values * 100
)

plt.title("Subscription Rate by Education")
plt.xlabel("Education")
plt.ylabel("Subscription Rate (%)")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================================
# 6. REMOVE DATA LEAKAGE
# ============================================================

# Duration represents the length of the marketing call.
# It is known only after the call.
# Therefore, we remove it to make the prediction realistic.

if "duration" in df.columns:
    df = df.drop(columns=["duration"])


# ============================================================
# 7. FEATURES AND TARGET
# ============================================================

X = df.drop("y", axis=1)

y = df["y"]


# ============================================================
# 8. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# 10. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 11. DECISION TREE MODEL
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ============================================================
# 12. HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "classifier__max_depth": [4, 6, 8],
    "classifier__min_samples_split": [2, 10, 20],
    "classifier__min_samples_leaf": [1, 5, 10],
    "classifier__criterion": ["gini", "entropy"]
}

print("\nTraining Decision Tree...")
print("Please wait...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 13. BEST MODEL
# ============================================================

model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(
    round(grid_search.best_score_, 4)
)


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Subscription",
            "Subscription"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Subscription",
        "Subscription"
    ]
)

disp.plot()

plt.title("Decision Tree - Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 18. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title("Decision Tree - ROC Curve")

plt.tight_layout()
plt.show()


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

trained_preprocessor = model.named_steps[
    "preprocessor"
]

trained_tree = model.named_steps[
    "classifier"
]

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": trained_tree.feature_importances_
})

importance = (
    importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .head(15)
)

print("\nTop 15 Important Features:")
print(importance)


# Feature importance graph

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"][::-1],
    importance["Importance"][::-1]
)

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ============================================================
# 20. DECISION TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(24, 12))

plot_tree(
    trained_tree,
    feature_names=feature_names,
    class_names=[
        "No Subscription",
        "Subscription"
    ],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title(
    "Bank Marketing Decision Tree - First 3 Levels"
)

plt.tight_layout()
plt.show()


# ============================================================
# 21. SAMPLE CUSTOMER PREDICTION
# ============================================================

sample_customer = X_test.iloc[[0]]

prediction = model.predict(
    sample_customer
)[0]

probability = model.predict_proba(
    sample_customer
)[0][1]

print("\n" + "=" * 60)
print("SAMPLE CUSTOMER PREDICTION")
print("=" * 60)

if prediction == 1:
    print("Prediction: Customer is likely to subscribe.")
else:
    print("Prediction: Customer is unlikely to subscribe.")

print(
    f"Subscription Probability: "
    f"{probability * 100:.2f}%"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 03 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFinal Model Metrics:")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nTop 10 Important Features:")

for _, row in importance.head(10).iterrows():
    print(
        f"{row['Feature']} : "
        f"{row['Importance']:.4f}"
    )
    