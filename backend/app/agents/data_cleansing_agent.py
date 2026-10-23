"""
Dedicated Scientific Data Cleansing, Schema Curation & Leakage Pruning Agent (`data-cleansing-agent`).
Responsible for:
1. Multi-format tabular ingestion (CSV, TSV, Excel .xlsx/.xls, Parquet, JSON) with magic-byte verification,
   encoding fallback (`utf-8-sig`, `utf-8`, `latin1`, `cp1252`), and delimiter auto-detection (`,`, `;`, `\t`, `|`).
2. Title-banner row detection & header promotion when initial headers are `Unnamed:*`.
3. Empty row/column removal, duplicate header deduplication, and `Unnamed` index stripping vs. data recovery.
4. Scientific measurement string coercion (`12.4 ± 1.2`, `84.2%`, `0.85 (0.78-0.91)`, `<0.001**`, `1,234.5` -> `float64`).
5. Identifier/junk column filtering and structured cleansing audit reporting (`cleansing_report`).
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.agents.pdf_extractor import parse_scientific_number
from app.config import settings

logger = logging.getLogger(__name__)

NON_TARGET_METADATA_NAMES = {
    "id",
    "index",
    "row",
    "row_id",
    "sample_id",
    "patient_id",
    "subject_id",
    "nct_id",
    "concept_id",
    "cell_id",
    "orig.ident",
    "uuid",
    "run_id",
    "no",
    "number",
    "serial",
    "batch",
    "split",
    "fold",
    "train",
    "test",
    "dataset",
    "cohort_id",
    "site_id",
    "center_id",
    "statement",
    "description",
    "comment",
    "notes",
    "url",
    "uri",
    "file",
    "filepath",
    "filename",
    "timestamp",
    "date",
    "start_index",
    "end_index",
    "x_slide_mm",
    "y_slide_mm",
}


def is_string_or_object_series(series: pd.Series) -> bool:
    """Safely detects string/object columns across Pandas 1.x, 2.x, and 3.x StringDtype."""
    return bool(
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or (
            not pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
            and not isinstance(series.dtype, pd.CategoricalDtype)
        )
    )


def is_junk_or_id_column(col_name: str, series: pd.Series, n_rows: int) -> bool:
    """
    Identifies non-predictive identifier columns (`Unnamed: 0`, monotonic row indices, sample IDs, UUIDs, free-text)
    so they are excluded from modeling targets and features.
    """
    cl = str(col_name).strip().lower()
    if (
        cl.startswith("unnamed")
        or cl in NON_TARGET_METADATA_NAMES
        or cl.endswith("_id")
        or cl.startswith("id_")
        or cl.endswith("_text")
        or cl.endswith("_url")
        or cl.endswith("_name")
        or re.match(r"^(sample_?id|subject_?id|patient_?id|donor_?id|row_?id|record_?id)[_\-\s]*\d*$", cl)
        or re.match(r"^\d+$", cl)
    ):
        return True

    # Check if numeric column is a monotonic 0..N-1 or 1..N row counter
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and n_rows > 5:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) == n_rows and clean.nunique() == n_rows:
            diffs = np.diff(clean.values)
            if np.all(diffs == 1) or np.all(diffs == -1):
                return True
        if len(clean) >= max(20, int(n_rows * 0.90)) and clean.nunique() == len(clean):
            idx_corr = abs(float(np.corrcoef(np.arange(len(clean)), clean.values)[0, 1]))
            if not np.isnan(idx_corr) and idx_corr >= 0.996:
                return True

    # Check if string/object column is a unique row identifier or long free-text note column
    if is_string_or_object_series(series):
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return True
        nunique = non_null.nunique()
        if n_rows >= 15 and nunique >= max(20, int(n_rows * 0.85)):
            return True
        avg_len = float(non_null.str.len().mean())
        if avg_len > 45.0:
            return True

    return False


class DataCleansingAgent:
    def _read_delimited_with_fallbacks(
        self, path: Path, preferred_sep: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str]:
        encodings = ("utf-8-sig", "utf-8", "latin1", "cp1252")
        candidate_seps = [preferred_sep] if preferred_sep else [",", ";", "\t", "|", None]
        best_df: Optional[pd.DataFrame] = None
        best_desc = "CSV (utf-8)"

        sep_labels = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe", None: "auto-sniffed"}

        for enc in encodings:
            for sep in candidate_seps:
                try:
                    if sep is None:
                        df = pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
                    else:
                        df = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip")
                    desc = f"Delimited ({sep_labels.get(sep, 'auto')}, {enc})"
                    if df is not None and len(df) >= 1 and df.shape[1] >= 2:
                        return df, desc
                    if best_df is None or (df is not None and df.shape[1] > best_df.shape[1]):
                        best_df = df
                        best_desc = desc
                except UnicodeDecodeError:
                    break
                except Exception:
                    continue

        if best_df is not None:
            return best_df, best_desc
        return pd.read_csv(path, encoding="latin1", on_bad_lines="skip"), "Delimited (latin1 fallback)"

    def _read_excel_with_fallbacks(self, path: Path) -> Tuple[pd.DataFrame, str]:
        try:
            xl = pd.ExcelFile(path)
            best_sheet_df = None
            best_sheet_name = xl.sheet_names[0] if xl.sheet_names else "Sheet1"
            for sname in xl.sheet_names:
                sdf = xl.parse(sname)
                if sdf.shape[1] >= 2 and len(sdf) >= 4:
                    best_sheet_df = sdf
                    best_sheet_name = sname
                    break
                if best_sheet_df is None or (sdf.shape[0] * sdf.shape[1] > best_sheet_df.shape[0] * best_sheet_df.shape[1]):
                    best_sheet_df = sdf
                    best_sheet_name = sname
            if best_sheet_df is not None:
                return best_sheet_df, f"Excel Workbook (sheet='{best_sheet_name}')"
        except Exception:
            pass
        try:
            return pd.read_excel(path), "Excel Workbook"
        except Exception:
            return self._read_delimited_with_fallbacks(path, preferred_sep=None)

    def cleanse_tabular_file(self, file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads and cleanses a tabular dataset file (.csv, .tsv, .xlsx, .xls, .parquet, .json),
        returning (clean_df, cleansing_report).
        """
        path = Path(file_path)
        if not path.exists():
            fallback = settings.DATASET_DIR / path.name
            if fallback.exists():
                path = fallback

        suffix = path.suffix.lower()
        magic = b""
        try:
            with open(path, "rb") as fh:
                magic = fh.read(8)
        except Exception:
            pass

        if magic[:4] == b"PK\x03\x04" or magic[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" or suffix in (".xlsx", ".xls"):
            df, detected_format = self._read_excel_with_fallbacks(path)
        elif magic[:4] == b"PAR1" or suffix == ".parquet":
            df = pd.read_parquet(path)
            detected_format = "Apache Parquet"
        elif suffix in (".tsv", ".tab"):
            df, detected_format = self._read_delimited_with_fallbacks(path, preferred_sep="\t")
        elif suffix == ".json":
            df = pd.read_json(path)
            detected_format = "JSON Tabular"
        else:
            df, detected_format = self._read_delimited_with_fallbacks(path, preferred_sep=None)

        if df is None or df.empty:
            return pd.DataFrame(), {
                "agent": "data-cleansing-agent",
                "detected_format": detected_format,
                "raw_shape": [0, 0],
                "cleaned_shape": [0, 0],
                "actions": ["Empty or unreadable dataset."],
            }

        raw_rows, raw_cols = int(df.shape[0]), int(df.shape[1])
        actions_log: List[str] = [f"Ingested {detected_format} ({raw_rows} rows × {raw_cols} cols)."]
        header_promoted_from_row: Optional[int] = None
        dropped_columns: List[Dict[str, str]] = []
        renamed_columns: List[Dict[str, str]] = []
        scientific_coercions: List[Dict[str, Any]] = []

        # 1. Strip completely empty rows
        df = df.dropna(how="all").reset_index(drop=True)
        empty_rows_dropped = raw_rows - len(df)
        if empty_rows_dropped > 0:
            actions_log.append(f"Dropped {empty_rows_dropped} completely empty row(s).")

        # 2. Detect title-banner rows preceding the real column header row
        unnamed_cnt = sum(
            1 for c in df.columns if str(c).strip().lower().startswith("unnamed") or str(c).strip() == ""
        )
        if df.shape[1] >= 2 and unnamed_cnt >= max(1, int(df.shape[1] * 0.5)) and len(df) >= 4:
            for r_idx in range(min(6, len(df) - 2)):
                row_vals = [str(v).strip() if pd.notna(v) else "" for v in df.iloc[r_idx].values]
                non_empty = [v for v in row_vals if v and v.lower() != "nan"]
                num_in_row = sum(1 for v in non_empty if parse_scientific_number(v) is not None)
                if len(non_empty) >= max(2, int(df.shape[1] * 0.5)) and num_in_row < len(non_empty) * 0.5:
                    new_cols = []
                    seen_c: Dict[str, int] = {}
                    for c_i, val in enumerate(row_vals):
                        cname = val if val and val.lower() != "nan" else f"Variable_{c_i + 1}"
                        if cname in seen_c:
                            seen_c[cname] += 1
                            cname = f"{cname}_{seen_c[cname]}"
                        else:
                            seen_c[cname] = 1
                        new_cols.append(cname)
                    df = df.iloc[r_idx + 1 :].copy().reset_index(drop=True)
                    df.columns = new_cols
                    header_promoted_from_row = int(r_idx + 1)
                    actions_log.append(
                        f"Promoted true column headers from row {r_idx + 1} (stripped title banner row)."
                    )
                    break

        # 3. Handle remaining `Unnamed:*` columns (drop index/empty columns; rename data-bearing columns)
        cols_to_drop = []
        rename_map = {}
        for idx, c in enumerate(df.columns):
            c_str = str(c).strip()
            if c_str.lower().startswith("unnamed") or c_str == "":
                s = df.iloc[:, idx]
                if float(s.isna().mean()) >= 0.70:
                    cols_to_drop.append(c)
                    dropped_columns.append({"column": c_str or f"col_{idx}", "reason": "Empty unnamed column (>=70% NaN)"})
                elif idx == 0 and is_junk_or_id_column(c_str, s, len(df)):
                    cols_to_drop.append(c)
                    dropped_columns.append({"column": c_str or "Unnamed: 0", "reason": "Monotonic row index column"})
                else:
                    new_name = f"Variable_{idx + 1}"
                    rename_map[c] = new_name
                    renamed_columns.append({"from": c_str or f"col_{idx}", "to": new_name})

        if cols_to_drop and len(df.columns) - len(cols_to_drop) >= 2:
            df = df.drop(columns=cols_to_drop)
            actions_log.append(f"Dropped {len(cols_to_drop)} index/empty column(s): {[d['column'] for d in dropped_columns]}.")
        if rename_map:
            df = df.rename(columns=rename_map)
            actions_log.append(f"Recovered {len(rename_map)} unnamed data column(s) as named variables.")

        # 4. Deduplicate repeated column names
        if len(set(df.columns)) < len(df.columns):
            deduped = []
            seen_names: Dict[str, int] = {}
            for c in df.columns:
                cs = str(c).strip()
                if cs in seen_names:
                    seen_names[cs] += 1
                    new_c = f"{cs}_{seen_names[cs]}"
                    deduped.append(new_c)
                    renamed_columns.append({"from": cs, "to": new_c})
                else:
                    seen_names[cs] = 1
                    deduped.append(cs)
            df.columns = deduped
            actions_log.append("Deduplicated repeated column headers.")

        # 5. Coerce scientific numeric strings (`12.4 ± 1.2`, `84.2%`, `0.85 (0.78-0.91)`, `<0.001**`, `1,234.5`) into floats
        for col in df.columns:
            s = df[col]
            if is_string_or_object_series(s):
                non_null = s.dropna().astype(str).str.strip()
                non_null = non_null[non_null != ""]
                if len(non_null) >= 3:
                    parsed = s.map(parse_scientific_number)
                    ratio = float(parsed.notna().sum()) / max(1, len(non_null))
                    if ratio >= 0.55:
                        sample_raw = str(non_null.iloc[0])[:28]
                        sample_num = float(parsed.dropna().iloc[0])
                        df[col] = parsed.astype(float)
                        scientific_coercions.append(
                            {
                                "column": str(col),
                                "valid_ratio": round(ratio, 3),
                                "example": f"'{sample_raw}' -> {sample_num}",
                            }
                        )
        if scientific_coercions:
            actions_log.append(
                f"Coerced {len(scientific_coercions)} scientific string column(s) (±, %, CI, inequalities) to float64: "
                f"{[c['column'] for c in scientific_coercions]}."
            )

        # 6. Replace ±inf with NaN for safe inside-fold median imputation
        num_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = 0
        if len(num_cols) > 0:
            inf_mask = np.isinf(df[num_cols].values)
            inf_count = int(np.sum(inf_mask))
            if inf_count > 0:
                df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
                actions_log.append(f"Replaced {inf_count} infinite (±inf) cell(s) with NaN for inside-fold imputation.")

        total_missing = int(df.isna().sum().sum())
        cols_with_missing = int((df.isna().sum() > 0).sum())
        if total_missing > 0:
            actions_log.append(
                f"Detected {total_missing} missing cell(s) across {cols_with_missing} column(s); routed to inside-fold median/ordinal imputation."
            )
        else:
            actions_log.append("Verified 0 missing values across all retained columns.")

        report: Dict[str, Any] = {
            "agent": "data-cleansing-agent",
            "skill": "scientific-data-cleanser",
            "detected_format": detected_format,
            "raw_shape": [raw_rows, raw_cols],
            "cleaned_shape": [int(df.shape[0]), int(df.shape[1])],
            "header_promoted_from_row": header_promoted_from_row,
            "empty_rows_dropped": int(empty_rows_dropped),
            "dropped_columns": dropped_columns,
            "renamed_columns": renamed_columns,
            "scientific_coercions": scientific_coercions,
            "inf_values_repaired": inf_count,
            "total_missing_cells": total_missing,
            "columns_with_missing": cols_with_missing,
            "imputation_policy": "Inside-fold SimpleImputer(median) + unknown-safe OrdinalEncoder fit strictly on X_train",
            "actions": actions_log,
        }
        return df, report


data_cleansing_agent = DataCleansingAgent()
