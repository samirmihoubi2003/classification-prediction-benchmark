from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class PreprocessSpec:
    numeric: List[str]
    categorical: List[str]


def build_preprocess(X: pd.DataFrame) -> PreprocessSpec:
    numeric = X.select_dtypes(include=["number", "int64", "float64"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return PreprocessSpec(numeric=numeric, categorical=categorical)


def make_preprocessor(spec: PreprocessSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), spec.numeric),
            (
                "cat",
                Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                spec.categorical,
            ),
        ],
        remainder="drop",
    )


def get_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=2000),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "svm": SVC(kernel="rbf", C=2.0, gamma="scale"),
        "dt": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=400, random_state=random_state, n_jobs=-1),
    }


def make_pipeline(preprocessor: ColumnTransformer, model: object) -> Pipeline:
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])
