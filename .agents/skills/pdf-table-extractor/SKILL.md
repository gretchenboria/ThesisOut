---
name: pdf-table-extractor
description: Use this skill when parsing uploaded dissertation PDFs to extract bibliographic metadata, baseline statistical methods, repository DOIs, and embedded quantitative tables without synthetic fallback data.
---

# PDF Dissertation & Quantitative Table Extractor Skill

## Overview
Extracts real bibliographic metadata, empirical claims, repository links (Zenodo, Figshare, Dryad, OSF, GitHub, NCBI GEO), and tabular datasets from uploaded PDF files.

## Execution Steps
1. Extract text and metadata across pages using `pypdf.PdfReader`.
2. Scan for embedded repository DOIs/URLs (`10.5281/zenodo.*`, `zenodo.org/records/*`). If present, fetch the real tabular dataset via `harvester_agent.fetch_zenodo_record_by_id`.
3. Extract embedded PDF tables using `pdfplumber`:
   - Pass 1: Grid-line table extraction (`page.extract_tables()`).
   - Pass 2: Text-aligned borderless table extraction (`page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})`).
   - Pass 3: Whitespace/tab-aligned numeric row parser on extracted page text.
4. Validate that at least 2 columns coerce to numeric (`>= 55%` non-null) and save to `storage/datasets/`. Never generate synthetic fallback data.
