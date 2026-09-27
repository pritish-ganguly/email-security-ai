from src.data.spamassassin_loader import load_spamassassin
from src.data.dataset_splitter import split_dataset
from src.preprocessing.cleaner import clean_dataset
from src.features.text_features import (
    fit_text_pipeline,
    transform_text,
    print_text_feature_summary,
)
from src.models.supervised_baseline import (
    create_models,
    train_models,
)
from src.evaluation.evaluation import (
    evaluate_model,
    analyze_thresholds,
    print_threshold_summary,
    compare_results,
)


def main():
    print("\nLoading SpamAssassin dataset...")

    dataframe = load_spamassassin()

    print("\nCleaning dataset...")

    dataframe, _ = clean_dataset(
        dataframe
    )

    print("\nCreating leakage-safe dataset split...")

    split = split_dataset(
        dataframe
    )

    train_data = split.train
    validation_data = split.validation
    test_data = split.test

    x_train_text = train_data["email_text"]
    x_validation_text = validation_data["email_text"]
    x_test_text = test_data["email_text"]

    y_train = train_data["label"]
    y_validation = validation_data["label"]
    y_test = test_data["label"]

    print(
        "\nFitting NLP feature pipeline "
        "on training data only..."
    )

    text_pipeline = fit_text_pipeline(
        x_train_text
    )

    print("\nTransforming training data...")

    x_train = transform_text(
        text_pipeline,
        x_train_text,
    )

    print("Transforming validation data...")

    x_validation = transform_text(
        text_pipeline,
        x_validation_text,
    )

    print("Transforming test data...")

    x_test = transform_text(
        text_pipeline,
        x_test_text,
    )

    print_text_feature_summary(
        text_pipeline,
        x_train,
    )

    print("\n" + "=" * 70)
    print("FEATURE MATRIX SHAPES")
    print("=" * 70)

    print(
        f"\nTrain:      {x_train.shape}"
    )

    print(
        f"Validation: {x_validation.shape}"
    )

    print(
        f"Test:       {x_test.shape}"
    )

    print("\nCreating supervised models...")

    models = create_models()

    print("\nTraining models...")

    trained_models = train_models(
        models,
        x_train,
        y_train,
    )

    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)

    evaluation_results = []

    for name, model in trained_models.items():
        result = evaluate_model(
            model=model,
            name=name,
            features=x_validation,
            labels=y_validation,
        )

        evaluation_results.append(
            result
        )

    compare_results(
        evaluation_results
    )

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    for name, model in trained_models.items():
        print(
            f"\nAnalyzing: {name}"
        )

        try:
            threshold_results = analyze_thresholds(
                model=model,
                features=x_validation,
                labels=y_validation,
            )

            print_threshold_summary(
                threshold_results
            )

        except ValueError as error:
            print(
                f"Threshold analysis skipped: "
                f"{error}"
            )

    print("\n" + "=" * 70)
    print("TEST SET STATUS")
    print("=" * 70)

    print(
        "\nThe test set has been transformed "
        "but has NOT been used for model "
        "selection or threshold selection."
    )

    print(
        f"\nTest records: {len(y_test)}"
    )


if __name__ == "__main__":
    main()