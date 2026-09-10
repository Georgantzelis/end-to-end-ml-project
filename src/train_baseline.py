from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_data
from src.preprocessing import prepare_features_and_target


RANDOM_STATE = 42


def main():
    df = load_data()
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    print("\nAccuracy:",
          round(accuracy_score(y_test, predictions), 4))
    print("Precision:",
          round(precision_score(y_test, predictions), 4))
    print("Recall:",
          round(recall_score(y_test, predictions), 4))
    print("F1:",
          round(f1_score(y_test, predictions), 4))
    print("ROC-AUC:",
          round(roc_auc_score(y_test, probabilities), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()