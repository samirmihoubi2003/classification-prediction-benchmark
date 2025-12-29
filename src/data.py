
### `src/data.py`
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

BANK_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"


@dataclass(frozen=True)
class DatasetPaths:
    raw: Path = Path("data/raw")

    def ensure(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)


def download_and_extract_bank(paths: DatasetPaths) -> Path:
    """
    Download bank-additional.zip and extract bank-additional-full.csv
    """
    paths.ensure()
    zip_path = paths.raw / "bank-additional.zip"
    if not zip_path.exists():
        print(f"[download] {BANK_ZIP_URL} -> {zip_path}")
        urllib.request.urlretrieve(BANK_ZIP_URL, zip_path)

    out_csv = paths.raw / "bank-additional-full.csv"
    if not out_csv.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            member = "bank-additional/bank-additional-full.csv"
            print(f"[extract] {member} -> {out_csv}")
            with zf.open(member) as src, open(out_csv, "wb") as dst:
                dst.write(src.read())

    return out_csv


def load_bank_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df["y"] = (df["y"].astype(str).str.lower() == "yes").astype(int)
    return df


def train_test_split_bank(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["y"])
    y = df["y"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
