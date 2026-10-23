---
name: scientific-data-cleanser
description: Use this skill when cleansing raw scientific datasets (.csv, .tsv, .xlsx, .xls, .parquet, or extracted PDF tables), recovering headers from title-banner rows, coercing scientific measurement strings (mean ± SD, %, CI brackets, inequalities) into floats, stripping identifier and target-leakage columns, and performing inside-fold imputation and encoding.
---

# Scientific Data Cleanser Skill (`scientific-data-cleanser`)

## Cleansing Pipeline Stages
1. **Stage 1 — Ingestion & Format Sniffing (`cleanse_tabular_file`)**:
   - Inspects magic bytes and file extensions to route `.xlsx`, `.xls`, `.parquet`, `.json`, `.tsv`, and `.csv` files.
   - Tests `utf-8-sig`, `utf-8`, `latin1`, and `cp1252` encodings and `,`, `;`, `\t`, and `|` delimiters.
   - Scans multi-sheet Excel workbooks to select the worksheet with valid tabular dimensions ($\ge 4$ rows, $\ge 2$ columns).
2. **Stage 2 — Header Promotion & Column Sanitization**:
   - Promotes the true column header row when row 0 is a title banner ($\ge 50\%$ `Unnamed:*` columns).
   - Drops empty rows/columns and monotonic integer index columns (`Unnamed: 0`), deduplicating any repeated column headers.
3. **Stage 3 — Scientific Cell Coercion (`parse_scientific_number`)**:
   - Extracts numeric measurements from strings formatted as `12.4 ± 1.2`, `84.2%`, `0.85 [0.78–0.91]`, `<0.001**`, and `1,234.5` when $\ge 55\%$ of non-null values in a column represent numeric measurements.
4. **Stage 4 — Leakage Pruning & Inside-Fold Imputation**:
   - Removes sample IDs, free-text notes, boolean complements, and collinear leakage proxies on `X_train`.
   - Fits median imputation and `OrdinalEncoder` strictly on training folds.
