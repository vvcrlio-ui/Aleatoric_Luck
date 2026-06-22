"""Preprocessing for tabular FFCWS challenge features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_MISSING_CODES = [-9, -8, -7, -6, -5, -4, -3, -2, -1]


@dataclass
class Preprocessor:
    missing_codes: list[Any] = field(default_factory=lambda: DEFAULT_MISSING_CODES.copy())
    missing_threshold: float = 0.8
    low_variance_threshold: float = 1e-12
    standardize: bool = False
    imputation_strategy: str = "median"

    retained_columns_: list[str] = field(default_factory=list, init=False)
    numeric_columns_: list[str] = field(default_factory=list, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    categorical_levels_: dict[str, list[str]] = field(default_factory=dict, init=False)
    medians_: dict[str, float] = field(default_factory=dict, init=False)
    missing_indicator_columns_: list[str] = field(default_factory=list, init=False)
    dropped_: dict[str, list[str]] = field(default_factory=dict, init=False)
    scaler_: StandardScaler | None = field(default=None, init=False)
    output_columns_: list[str] = field(default_factory=list, init=False)

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = X.copy()
        for code in self.missing_codes:
            cleaned = cleaned.replace(code, np.nan)
            cleaned = cleaned.replace(str(code), np.nan)
        cleaned = cleaned.replace({"NA": np.nan, "NaN": np.nan, "": np.nan})
        return cleaned

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        if self.imputation_strategy not in {"median", "native_nan"}:
            raise ValueError(
                "imputation_strategy must be one of: median, native_nan"
            )
        if self.standardize and self.imputation_strategy == "native_nan":
            raise ValueError("native_nan preprocessing cannot be standardized")

        cleaned = self._clean(X)
        missing_rate = cleaned.isna().mean()
        all_empty = missing_rate[missing_rate >= 1.0].index.tolist()
        high_missing = missing_rate[
            (missing_rate >= self.missing_threshold) & (missing_rate < 1.0)
        ].index.tolist()

        candidates = [
            col for col in cleaned.columns if col not in set(all_empty + high_missing)
        ]
        low_var = []
        for col in candidates:
            series = cleaned[col].dropna()
            if series.empty or series.nunique(dropna=True) <= 1:
                low_var.append(col)
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() > 0.95 and float(numeric.var()) <= self.low_variance_threshold:
                low_var.append(col)

        self.retained_columns_ = [
            col for col in candidates if col not in set(low_var)
        ]
        retained = cleaned[self.retained_columns_]
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        for col in self.retained_columns_:
            numeric = pd.to_numeric(retained[col], errors="coerce")
            if retained[col].dtype == object and numeric.notna().mean() < 0.95:
                self.categorical_columns_.append(col)
            else:
                self.numeric_columns_.append(col)
                self.medians_[col] = float(numeric.median()) if numeric.notna().any() else 0.0

        self.categorical_levels_ = {}
        for col in self.categorical_columns_:
            values = retained[col].astype("object")
            if self.imputation_strategy == "native_nan":
                values = values.dropna()
            else:
                values = values.where(values.notna(), "__MISSING__")
            self.categorical_levels_[col] = sorted(
                values.astype(str).unique().tolist()
            )
        if self.imputation_strategy == "median":
            self.missing_indicator_columns_ = [
                col for col in self.retained_columns_ if retained[col].isna().any()
            ]
        else:
            self.missing_indicator_columns_ = []

        transformed = self._transform_no_scale(X)
        self.output_columns_ = transformed.columns.tolist()
        if self.standardize:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(transformed)
        self.dropped_ = {
            "all_empty": all_empty,
            "high_missing": high_missing,
            "low_variance": low_var,
        }
        return self

    def _transform_no_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = self._clean(X)
        parts: list[pd.DataFrame] = []

        if self.imputation_strategy == "median":
            indicators = {
                f"{col}__missing": cleaned[col].isna().astype(float)
                for col in self.missing_indicator_columns_
            }
            if indicators:
                parts.append(pd.DataFrame(indicators, index=cleaned.index))

        if self.numeric_columns_:
            if self.imputation_strategy == "native_nan":
                numeric = pd.DataFrame(
                    {
                        col: pd.to_numeric(cleaned[col], errors="coerce")
                        for col in self.numeric_columns_
                    },
                    index=cleaned.index,
                )
            else:
                numeric = pd.DataFrame(
                    {
                        col: pd.to_numeric(cleaned[col], errors="coerce").fillna(
                            self.medians_[col]
                        )
                        for col in self.numeric_columns_
                    },
                    index=cleaned.index,
                )
            parts.append(numeric)

        for col in self.categorical_columns_:
            if self.imputation_strategy == "native_nan":
                values = cleaned[col].astype("object").where(cleaned[col].notna(), np.nan)
            else:
                values = cleaned[col].astype("object").where(cleaned[col].notna(), "__MISSING__")
            values = values.astype(str)
            dummies = pd.get_dummies(values, prefix=col, dtype=float)
            for level in self.categorical_levels_[col]:
                name = f"{col}_{level}"
                if name not in dummies.columns:
                    dummies[name] = 0.0
            parts.append(dummies[[f"{col}_{level}" for level in self.categorical_levels_[col]]])

        if not parts:
            return pd.DataFrame(index=cleaned.index)
        return pd.concat(parts, axis=1).copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_no_scale(X)
        transformed = transformed.reindex(columns=self.output_columns_, fill_value=0.0)
        if self.standardize and self.scaler_ is not None:
            values = self.scaler_.transform(transformed)
            return pd.DataFrame(values, columns=self.output_columns_, index=transformed.index)
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def report(self) -> dict[str, Any]:
        return {
            "retained_columns": self.retained_columns_,
            "numeric_columns": self.numeric_columns_,
            "categorical_columns": self.categorical_columns_,
            "missing_indicator_columns": self.missing_indicator_columns_,
            "dropped": self.dropped_,
            "n_output_columns": len(self.output_columns_),
            "standardize": self.standardize,
            "imputation_strategy": self.imputation_strategy,
        }
