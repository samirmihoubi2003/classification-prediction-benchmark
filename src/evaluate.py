from __future__ import annotations

from pathlib import Path
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def run_cross_validation(estimator, X, y, seed: int = 42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = ["accuracy", "precision", "recall", "f1"]
    return cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)


def summarize_cv(cv_out: dict) -> dict:
    summary = {}
    for k, v in cv_out.items():
        if k.startswith("test_"):
            metric = k.replace("test_", "")
            summary[f"{metric}_mean"] = float(np.mean(v))
            summary[f"{metric}_std"] = float(np.std(v))
    return summary


def save_classification_report(estimator, X_test, y_test, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = estimator.predict(X_test)
    report_txt = classification_report(y_test, y_pred, digits=4)
    (out_dir / f"{name}.txt").write_text(report_txt, encoding="utf-8")


def save_confusion_matrix_plot(estimator, X_test, y_test, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig = plt.figure()
    ax = plt.gca()
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=160)
    plt.close(fig)
