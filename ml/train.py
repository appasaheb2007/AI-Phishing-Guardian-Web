"""
AI Phishing Guardian
Train the phishing URL detector using the PhiUSIIL dataset.

PhiUSIIL label convention:
    1 = legitimate
    0 = phishing

Our application convention:
    0 = legitimate
    1 = phishing
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from detector import features


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = ROOT / "ml" / "model.joblib"


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print("       AI PHISHING GUARDIAN - MODEL TRAINING")
    print("=" * 65)
    print()

    print(f"Dataset : {DATASET_PATH}")
    print(f"Model   : {MODEL_PATH}")
    print()

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "\nDataset not found.\n\n"
            f"Expected location:\n{DATASET_PATH}\n\n"
            "Make sure PhiUSIIL_Phishing_URL_Dataset.csv "
            "is inside the data folder."
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading PhiUSIIL dataset...")
    print()

    df = pd.read_csv(DATASET_PATH)

    print(f"Original dataset size: {len(df):,}")
    print()

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = {"URL", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            f"{sorted(missing_columns)}\n\n"
            "Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Keep URL and label
    # --------------------------------------------------------

    df = df[["URL", "label"]].copy()

    # Remove missing values
    df = df.dropna()

    # Convert URL to string
    df["URL"] = df["URL"].astype(str).str.strip()

    # Remove empty URLs
    df = df[df["URL"] != ""]

    # Remove duplicate URLs
    df = df.drop_duplicates(subset=["URL"])

    print(f"Unique usable URLs: {len(df):,}")
    print()

    # --------------------------------------------------------
    # Convert PhiUSIIL labels
    # --------------------------------------------------------
    #
    # PhiUSIIL:
    #     1 = legitimate
    #     0 = phishing
    #
    # Our application:
    #     0 = legitimate
    #     1 = phishing
    #
    # Therefore:
    #     PhiUSIIL 0 -> application 1
    #     PhiUSIIL 1 -> application 0
    # --------------------------------------------------------

    df["phishing_label"] = (
        df["label"].astype(int) == 0
    ).astype(int)

    legitimate_count = (
        df["phishing_label"] == 0
    ).sum()

    phishing_count = (
        df["phishing_label"] == 1
    ).sum()

    print("Dataset distribution:")
    print(f"Legitimate URLs : {legitimate_count:,}")
    print(f"Phishing URLs   : {phishing_count:,}")
    print()

    # --------------------------------------------------------
    # Extract URL features
    # --------------------------------------------------------

    print("Extracting URL features...")
    print("This may take some time for the large dataset.")
    print()

    first_features = features(
        df["URL"].iloc[0]
    )

    feature_names = [
        f"feature_{i}"
        for i in range(len(first_features))
    ]

    feature_rows = []

    for index, url in enumerate(df["URL"]):

        feature_rows.append(
            features(url)
        )

        # Progress every 10,000 URLs
        if (index + 1) % 10000 == 0:

            print(
                f"Processed {index + 1:,} / "
                f"{len(df):,} URLs"
            )

    X = pd.DataFrame(
        feature_rows,
        columns=feature_names
    )

    y = df["phishing_label"]

    print()
    print("Feature extraction complete.")
    print()

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    print("Creating training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")
    print()

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("=" * 65)
    print("TRAINING RANDOM FOREST")
    print("=" * 65)
    print()

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print("Model training completed.")
    print()

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("Testing model...")

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("                    TEST RESULTS")
    print("=" * 65)
    print()

    print(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-score  : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print()

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)
    print()

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "legitimate",
                "phishing"
            ],
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("=" * 65)
    print("CONFUSION MATRIX")
    print("=" * 65)
    print()

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(matrix)
    print()

    print(
        "Matrix format:"
    )

    print(
        "[[True Legitimate, False Phishing]"
    )

    print(
        " [False Legitimate, True Phishing]]"
    )

    print()

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print("Saving trained model...")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print()
    print("=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print()

    print(
        f"Model saved successfully:\n{MODEL_PATH}"
    )

    print()

    print("Your website can now use this model.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()