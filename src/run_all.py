from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data import DatasetPaths, download_and_extract_bank, load_bank_dataframe, train_test_split_bank
from src.modeling import build_preprocess, make_preprocessor, get_models, make_pipeline
from src.evaluate import (
    run_cross_validation,
    summarize_cv,
    save_classification_report,
    save_confusion_matrix_plot,
)


def main():
    results_dir = Path("results")
    reports_dir = results_dir / "test_reports"
    cms_dir = results_dir / "confusion_matrices"
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths()
    csv_path = download_and_extract_bank(paths)
    df = load_bank_dataframe(csv_path)

    X_train, X_test, y_train, y_test = train_test_split_bank(df)

    spec = build_preprocess(X_train)
    preprocessor = make_preprocessor(spec)

    rows = []
    models = get_models()

    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)

        print(f"\n=== Model: {name} ===")
        cv_out = run_cross_validation(pipe, X_train, y_train)
        summary = summarize_cv(cv_out)

        pipe.fit(X_train, y_train)

        save_classification_report(pipe, X_test, y_test, name, reports_dir)
        save_confusion_matrix_plot(pipe, X_test, y_test, name, cms_dir)

        rows.append({"model": name, **summary})
        print(summary)

    metrics = pd.DataFrame(rows).sort_values(by="f1_mean", ascending=False)
    metrics.to_csv(results_dir / "metrics_cv.csv", index=False)
    print("\n[done] results/metrics_cv.csv generated")
    print("[done] Check results/ folder")


if __name__ == "__main__":
    main()
