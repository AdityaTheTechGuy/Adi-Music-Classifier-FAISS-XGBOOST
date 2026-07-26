import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from data_pipeline import run_data_pipeline
from engine import (
    train_tuned_xgboost,
    get_hybrid_recommendations,
    build_faiss_index
)


MODEL_PATH = "xgboost_genre_artifacts.joblib"

# Set True while testing the new genre mapping/model.
# After you're happy with the trained model, change this to False.
FORCE_RETRAIN = True


def evaluate_model(
    classifier,
    X_test,
    y_test,
    y_train,
    label_encoder
):
    """
    Evaluate XGBoost using metrics that remain meaningful
    when the genre classes are imbalanced.
    """

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = classifier.predict(X_test)


    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )


    print(f"\nAccuracy:    {accuracy * 100:.2f}%")
    print(f"Macro F1:    {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")


    # --------------------------------------------------------
    # Majority-class baseline
    # --------------------------------------------------------

    majority_class = np.bincount(
        y_train
    ).argmax()

    baseline_predictions = np.full(
        shape=len(y_test),
        fill_value=majority_class
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )


    print("\n========================================")
    print("MAJORITY BASELINE")
    print("========================================")

    majority_name = label_encoder.inverse_transform(
        [majority_class]
    )[0]

    print(f"\nMajority class: {majority_name}")

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"XGBoost Accuracy:  "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Improvement:       "
        f"{(accuracy - baseline_accuracy) * 100:+.2f} percentage points"
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("========================================\n")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(
                len(label_encoder.classes_)
            ),
            target_names=label_encoder.classes_,
            digits=4,
            zero_division=0
        )
    )


    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "baseline_accuracy": baseline_accuracy
    }


def main():

    # ========================================================
    # STEP 1 — DATA PIPELINE
    # ========================================================

    print("\n--- Step 1: Executing Data Pipeline ---")

    df, df_cleaned, all_engineered_features = (
        run_data_pipeline("dataset.csv")
    )

    print(
        f"\nData pipeline complete. "
        f"Hybrid matrix shape: {df_cleaned.shape}\n"
    )


    # ========================================================
    # STEP 2 — PREPARE FEATURES
    # ========================================================

    print("--- Step 2: Preparing Features ---")

    best_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "loudness",
        "speechiness",
        "tempo",
        "valence"
    ]

    X = df_cleaned[best_features]

    y = df["macro_genre"]


    # ========================================================
    # STEP 3 — LOAD OR TRAIN MODEL
    # ========================================================

    should_train = (
        FORCE_RETRAIN
        or not os.path.exists(MODEL_PATH)
    )


    if not should_train:

        print(
            f"Found saved model artifacts at "
            f"'{MODEL_PATH}'. Loading..."
        )

        artifacts = joblib.load(
            MODEL_PATH
        )

        classifier = artifacts["model"]
        label_encoder = artifacts["label_encoder"]

        print(
            "Model and Label Encoder loaded successfully.\n"
        )


    else:

        if FORCE_RETRAIN and os.path.exists(MODEL_PATH):

            print(
                "FORCE_RETRAIN=True — ignoring existing "
                "model and training a new one."
            )

        else:

            print(
                "No saved model found. "
                "Initiating full training sequence..."
            )


        # ----------------------------------------------------
        # Encode macro genre labels
        # ----------------------------------------------------

        label_encoder = LabelEncoder()

        y_encoded = label_encoder.fit_transform(
            y
        )


        print(
            f"\nNumber of classes: "
            f"{len(label_encoder.classes_)}"
        )

        print("\nClasses:")

        for i, class_name in enumerate(
            label_encoder.classes_
        ):
            print(f"  {i}: {class_name}")


        # ----------------------------------------------------
        # Stratified 80/20 split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y_encoded,
                test_size=0.20,
                random_state=42,
                stratify=y_encoded
            )
        )


        print(
            f"\nTraining samples: {len(X_train):,}"
        )

        print(
            f"Testing samples:  {len(X_test):,}"
        )


        # ----------------------------------------------------
        # Balanced sample weights
        # ----------------------------------------------------

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )


        print(
            "\nBalanced sample weights calculated."
        )


        # ====================================================
        # TRAIN XGBOOST
        # ====================================================

        print(
            "\n--- Step 3: Training Model Engine ---"
        )


        # IMPORTANT:
        # train_tuned_xgboost() must accept sample_weight.
        classifier = train_tuned_xgboost(
            X_train,
            y_train,
        )


        # ====================================================
        # EVALUATE
        # ====================================================

        metrics = evaluate_model(
            classifier=classifier,
            X_test=X_test,
            y_test=y_test,
            y_train=y_train,
            label_encoder=label_encoder
        )


        # ====================================================
        # SAVE MODEL
        # ====================================================

        artifacts = {
            "model": classifier,
            "label_encoder": label_encoder,

            # Save metrics as metadata so we know how
            # this artifact performed later.
            "metrics": metrics,

            # Useful metadata for debugging/versioning.
            "classes": list(
                label_encoder.classes_
            ),

            "features": best_features
        }


        joblib.dump(
            artifacts,
            MODEL_PATH
        )


        print(
            f"\nArtifacts successfully saved to "
            f"'{MODEL_PATH}'"
        )


    # ========================================================
    # STEP 4 — FAISS RECOMMENDATION TEST
    # ========================================================

    print(
        "\n--- Step 4: Running Interactive "
        "Diversity Engine ---"
    )


    test_song = 'Kesariya (From "Brahmastra")'


    print(
        "Building FAISS index for local testing..."
    )


    faiss_index = build_faiss_index(
        df_cleaned,
        all_engineered_features
    )


    mock_rejections = []


    recs = get_hybrid_recommendations(
        song_name=test_song,
        df_meta=df,
        df_vect=df_cleaned,
        index=faiss_index,
        feature_cols=all_engineered_features,
        xgb_model=classifier,
        label_encoder=label_encoder,
        num_recs=5,
        rejected_ids=mock_rejections
    )


    # ========================================================
    # PRINT RECOMMENDATIONS
    # ========================================================

    if recs:

        print(
            "\n🔥 Standard Matches (Exploitation):"
        )

        for match in recs["standard_matches"]:

            print(
                f"[{match['index']}] "
                f"{match['track_name']} | "
                f"{match['artists']} | "
                f"{match['track_genre']} | "
                f"{match['similarity_score'] * 100:.1f}%"
            )


        print(
            "\n🎲 Diverse Matches (Exploration):"
        )

        for match in recs["diverse_matches"]:

            print(
                f"[{match['index']}] "
                f"{match['track_name']} | "
                f"{match['artists']} | "
                f"{match['track_genre']} | "
                f"{match['similarity_score'] * 100:.1f}%"
            )


    else:

        print("Song not found.")


if __name__ == "__main__":
    main()