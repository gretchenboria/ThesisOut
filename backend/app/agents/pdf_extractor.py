"""
PDF Dissertation Parser & In-PDF Table Extractor Agent.
1. Extracts text chapters, claims, baseline statistical methodology, and repository links (Zenodo, Figshare, Dryad, OSF, GitHub, GEO).
2. Uses `pdfplumber` to extract structured multi-row experimental/clinical/benchmark tables directly from dissertation PDFs
   and saves them as clean CSV datasets when external raw files are missing or supplementary.
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import pdfplumber
from pypdf import PdfReader

from app.config import settings

logger = logging.getLogger(__name__)

REPO_PATTERNS = [
    ("zenodo", re.compile(r"(?:https?://(?:www\.)?zenodo\.org/records?/\d+|10\.5281/zenodo\.\d+)", re.I)),
    ("figshare", re.compile(r"(?:https?://[^\s\)\"']*figshare\.com/[^\s\)\"']+|10\.6084/m9\.figshare\.\d+)", re.I)),
    ("dryad", re.compile(r"(?:https?://datadryad\.org/[^\s\)\"']+|10\.5061/dryad\.[^\s\)\"']+)", re.I)),
    ("osf", re.compile(r"https?://osf\.io/[a-z0-9]{5,}", re.I)),
    ("github", re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)),
    ("huggingface", re.compile(r"https?://huggingface\.co/datasets/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)),
    ("ncbi_geo", re.compile(r"\bGSE\d{4,8}\b")),
]


class PDFExtractorAgent:
    def extract_text_and_metadata(self, pdf_path: str | Path) -> Dict[str, Any]:
        path = Path(pdf_path)
        reader = PdfReader(str(path))
        full_text_chunks: List[str] = []
        max_pages = min(len(reader.pages), 60)
        for i in range(max_pages):
            try:
                page_txt = reader.pages[i].extract_text() or ""
                full_text_chunks.append(page_txt)
            except Exception as exc:
                logger.warning("Failed to read page %d of %s: %s", i, path, exc)

        full_text = "\n".join(full_text_chunks)
        first_page_text = full_text_chunks[0] if full_text_chunks else ""
        links = self.extract_repository_links(full_text)
        claims = self.extract_key_claims(full_text)
        stat_method = self.detect_author_statistical_method(full_text)
        parsed_meta = self.extract_bibliographic_metadata(first_page_text, path.stem)

        return {
            "page_count": len(reader.pages),
            "text_preview": full_text[:4000],
            "data_availability_links": links,
            "extracted_claims": claims,
            "author_statistical_method": stat_method,
            **parsed_meta,
        }

    def extract_bibliographic_metadata(self, first_page_text: str, fallback_stem: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in first_page_text.splitlines() if len(ln.strip()) > 3]
        title = ""
        university = "Research University"
        authors = "Dissertation Author"
        pub_year = 2024

        for ln in lines[:12]:
            low = ln.lower()
            if any(k in low for k in ["university", "institute of technology", "college", "school of", "eth ", "epfl"]):
                university = ln[:100]
                break

        for ln in lines[:8]:
            low = ln.lower()
            if not any(k in low for k in ["dissertation", "thesis", "submitted", "partial fulfillment", "university", "department", "degree of", "doctor of"]):
                if 12 <= len(ln) <= 180:
                    title = ln
                    break

        if not title:
            clean_stem = re.sub(r"^upload_\d+_", "", fallback_stem).replace("_", " ").replace("-", " ").strip()
            title = clean_stem.title() if clean_stem else "Uploaded Research Study"

        for i, ln in enumerate(lines[:20]):
            if ln.lower().startswith("by ") and len(ln) < 60:
                authors = ln[3:].strip()
                break
            if ln.lower() == "by" and i + 1 < len(lines) and len(lines[i + 1]) < 60:
                authors = lines[i + 1].strip()
                break

        year_matches = re.findall(r"\b(20[0-2]\d|199\d)\b", first_page_text)
        if year_matches:
            pub_year = int(year_matches[0])

        return {
            "parsed_title": title,
            "parsed_university": university,
            "parsed_authors": authors,
            "parsed_year": pub_year,
        }

    def extract_repository_links(self, text: str) -> List[Dict[str, str]]:
        found: List[Dict[str, str]] = []
        seen = set()
        for source_type, pattern in REPO_PATTERNS:
            for match in pattern.findall(text):
                cleaned = match.rstrip(".,;)")
                if cleaned not in seen:
                    seen.add(cleaned)
                    url = cleaned
                    if source_type == "zenodo" and cleaned.lower().startswith("10.5281/zenodo."):
                        rec_id = cleaned.split(".")[-1]
                        url = f"https://zenodo.org/records/{rec_id}"
                    elif source_type == "ncbi_geo" and cleaned.startswith("GSE"):
                        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={cleaned}"
                    found.append({"source_type": source_type, "identifier": cleaned, "url": url})
        return found

    def extract_key_claims(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claim_Triggers = (
            "we demonstrate",
            "our results show",
            "significantly associated",
            "we found that",
            "in contrast to previous",
            "unexpectedly",
            "future work",
            "limitation",
            "p <",
            "p =",
            "r2 =",
            "auc =",
        )
        claims: List[str] = []
        for s in sentences:
            clean_s = " ".join(s.split())
            if 55 <= len(clean_s) <= 320 and any(t in clean_s.lower() for t in claim_Triggers):
                claims.append(clean_s)
                if len(claims) >= 8:
                    break
        return claims or ["Primary empirical findings extracted from dissertation results section."]

    def detect_author_statistical_method(self, text: str) -> str:
        lower = text.lower()
        methods = []
        for label, kw in [
            ("Ordinary Least Squares (OLS) / Linear Regression", "linear regression"),
            ("Logistic Regression", "logistic regression"),
            ("Cox Proportional Hazards", "cox proportional"),
            ("One-Way / Two-Way ANOVA", "anova"),
            ("Student's t-test / Mann-Whitney U", "t-test"),
            ("Mixed-Effects Linear Model", "mixed-effects"),
            ("Kaplan-Meier Survival Analysis", "kaplan-meier"),
        ]:
            if kw in lower:
                methods.append(label)
        return ", ".join(methods[:2]) if methods else "Classical Parametric Linear Modeling (ANOVA / OLS)"

    def extract_tables_from_pdf(self, pdf_path: str | Path, dissertation_slug: str) -> List[Dict[str, Any]]:
        """
        Scans PDF pages for embedded quantitative data tables and reconstructs them as CSV files
        so LazyPredict can run even when external repository links are unavailable.
        """
        path = Path(pdf_path)
        extracted_artifacts: List[Dict[str, Any]] = []

        try:
            with pdfplumber.open(str(path)) as pdf:
                table_idx = 0
                for page_num, page in enumerate(pdf.pages[:50], start=1):
                    tables = page.extract_tables() or []
                    for raw_table in tables:
                        if not raw_table or len(raw_table) < 12:
                            continue
                        header = [str(c or f"col_{i}").strip().replace("\n", "_") for i, c in enumerate(raw_table[0])]
                        if len(header) < 3:
                            continue
                        rows = raw_table[1:]
                        df = pd.DataFrame(rows, columns=header)
                        # Attempt numeric coercion on columns
                        numeric_cols = 0
                        for col in df.columns:
                            coerced = pd.to_numeric(df[col], errors="coerce")
                            if coerced.notna().mean() >= 0.6:
                                df[col] = coerced
                                numeric_cols += 1

                        if numeric_cols >= 2 and len(df) >= 10:
                            table_idx += 1
                            out_name = f"{dissertation_slug}_pdf_table_p{page_num}_{table_idx}.csv"
                            out_path = settings.DATASET_DIR / out_name
                            df.to_csv(out_path, index=False)
                            extracted_artifacts.append(
                                {
                                    "name": f"Embedded PDF Cohort/Experimental Table (Page {page_num})",
                                    "source_type": "pdf_table",
                                    "source_url": f"file://{path}#page={page_num}",
                                    "file_path": str(out_path),
                                }
                            )
        except Exception as exc:
            logger.warning("PDF table extraction warning on %s: %s", path, exc)

        return extracted_artifacts


pdf_extractor_agent = PDFExtractorAgent()
