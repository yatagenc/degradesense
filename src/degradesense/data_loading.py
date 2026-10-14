"""Strict, read-only parsing of NASA C-MAPSS whitespace-delimited files."""

from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle", *SETTINGS, *SENSORS]


def project_root() -> Path:
    """Locate the source checkout without depending on the working directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Cannot locate project root; provide explicit file paths.")


def _read_numeric(path: Path, width: int) -> pd.DataFrame:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()  # tabs, repeated spaces and trailing whitespace
            if not fields:
                continue
            if len(fields) != width:
                raise ValueError(f"{path}:{line_number}: expected {width} columns, got {len(fields)}")
            try:
                row = [float(value) for value in fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric value") from exc
            if not np.isfinite(row).all():
                raise ValueError(f"{path}:{line_number}: missing or non-finite value")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return pd.DataFrame(rows)


def load_trajectories(path: Path) -> pd.DataFrame:
    """Validate schema, identifiers, uniqueness and per-engine cycle ordering.

    Rows are never silently sorted, deduplicated or imputed. Blank lines and
    whitespace-only empty fields are ignored; missing numeric tokens are errors.
    """
    frame = _read_numeric(path, len(COLUMNS))
    frame.columns = COLUMNS
    for column in ["unit_id", "cycle"]:
        values = frame[column]
        if ((values <= 0) | (values != np.floor(values))).any():
            raise ValueError(f"{column} must contain positive integers")
        frame[column] = values.astype("int64")
    if frame.duplicated(["unit_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle records")
    differences = frame.groupby("unit_id", sort=False)["cycle"].diff().dropna()
    if (differences <= 0).any():
        raise ValueError("Cycles must be strictly increasing within each engine")
    return frame


def load_rul(path: Path) -> pd.Series:
    """Load one nonnegative integer RUL per test engine, separately from features."""
    values = _read_numeric(path, 1).iloc[:, 0]
    if ((values < 0) | (values != np.floor(values))).any():
        raise ValueError("RUL must contain nonnegative integers")
    return pd.Series(values.to_numpy(dtype="int64"), name="rul")


def load_fd001(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_dir = Path(raw_dir) if raw_dir is not None else project_root() / "data" / "raw"
    train = load_trajectories(raw_dir / "train_FD001.txt")
    test = load_trajectories(raw_dir / "test_FD001.txt")
    rul = load_rul(raw_dir / "RUL_FD001.txt")
    ids = sorted(test.unit_id.unique())
    if ids != list(range(1, len(ids) + 1)) or len(rul) != len(ids):
        raise ValueError("RUL alignment requires one row per consecutive test unit ID starting at 1")
    rul.index = pd.Index(ids, name="unit_id")
    return train, test, rul
