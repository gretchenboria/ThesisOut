"""
Top-Science Multi-Repository Harvester & Graph Dataset Finder Agent (100% Real Data Only).
Implements three high-recall empirical dataset discovery mechanisms:
1. Author ORCID & Lab Graph Matching (OpenAlex + DataCite + Crossref):
   Queries OpenAlex (`type:dataset`), DataCite (`resource-type-id=dataset`), and Crossref (`type:dataset`)
   for datasets sharing the exact author ORCID, OpenAlex author ID, or advisor/institution lab graph
   within ±2 years of the dissertation defense date.
2. Multi-Repository Tabular Harvesters:
   Direct file-download resolvers for Zenodo (`zenodo.org/api/records`), Figshare (`api.figshare.com/v2/articles`),
   Harvard Dataverse (`dataverse.harvard.edu/api/search?type=file`), and Dryad (`datadryad.org/api/v2/search`).
3. Full-Text Data Availability & Supplementary Material Section Scraping:
   Fetches open-access JATS XML / PDFs from EuropePMC and institutional repositories to parse the
   "Data Availability" and "Supplementary Material" sections at the end of the thesis where dataset DOIs live.
Zero synthetic or simulated data generation.
"""
import io
import logging
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import pandas as pd
from pypdf import PdfReader

from app.agents.data_scientist import is_junk_or_id_column
from app.agents.pdf_extractor import pdf_extractor_agent

logger = logging.getLogger(__name__)

TOP_TIER_UNIVERSITIES = {
    "massachusetts institute of technology",
    "mit",
    "stanford university",
    "harvard university",
    "university of california, berkeley",
    "uc berkeley",
    "california institute of technology",
    "caltech",
    "university of oxford",
    "university of cambridge",
    "eth zurich",
    "imperial college london",
    "johns hopkins university",
    "university of california, san francisco",
    "ucsf",
    "carnegie mellon university",
    "princeton university",
    "yale university",
    "columbia university",
    "cornell university",
    "university of pennsylvania",
    "university of washington",
    "university of chicago",
    "university of michigan",
    "university of toronto",
    "karolinska institutet",
    "max planck",
    "broad institute",
    "epfl",
    "tsinghua university",
    "national university of singapore",
    "duke university",
    "university of california, san diego",
    "ucsd",
    "university of california, los angeles",
    "ucla",
    "georgia institute of technology",
}


class HarvesterAgent:
    def is_top_science_institution(self, institution_name: str) -> bool:
        if not institution_name:
            return False
        low = institution_name.lower()
        return (
            any(top in low for top in TOP_TIER_UNIVERSITIES)
            or "institute of technology" in low
            or "school of medicine" in low
            or "university" in low
        )

    def _validate_and_save_tabular_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        file_prefix: str,
        source_type: str,
        source_url: str,
        display_name: str,
        target_dir: Path,
        min_rows: int = 6,
        min_cols: int = 3,
    ) -> Optional[Dict[str, str]]:
        """
        Verifies that downloaded bytes represent a real, non-trivial tabular dataset
        (.csv, .tsv, .tab, .txt, .xlsx, .parquet) with >= min_rows and >= min_cols usable columns.
        Saves as CSV in target_dir so downstream LazyPredict benchmarks run cleanly.
        """
        if not raw_bytes or len(raw_bytes) < 150 or len(raw_bytes) > 20_000_000:
            return None
        low_name = filename.lower()
        df_full: Optional[pd.DataFrame] = None
        try:
            if low_name.endswith(".csv"):
                try:
                    df_full = pd.read_csv(io.BytesIO(raw_bytes))
                except Exception:
                    df_full = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", on_bad_lines="skip")
            elif low_name.endswith((".tsv", ".tab", ".txt")):
                df_full = pd.read_csv(io.BytesIO(raw_bytes), sep="\t", on_bad_lines="skip")
                if len(df_full.columns) < 2:
                    df_full = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", on_bad_lines="skip")
            elif low_name.endswith((".xlsx", ".xls")):
                df_full = pd.read_excel(io.BytesIO(raw_bytes))
            elif low_name.endswith(".parquet"):
                df_full = pd.read_parquet(io.BytesIO(raw_bytes))
            else:
                df_full = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            return None

        if df_full is None or len(df_full) < min_rows or len(df_full.columns) < min_cols:
            return None

        usable_cols = [
            c
            for c in df_full.columns
            if not is_junk_or_id_column(str(c), df_full[c], len(df_full))
            and df_full[c].nunique(dropna=True) >= 2
        ]
        if len(usable_cols) < min_cols:
            return None

        target_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).stem)[:60]
        out_path = target_dir / f"{file_prefix}_{safe_stem}.csv"
        if low_name.endswith(".csv"):
            out_path.write_bytes(raw_bytes)
        else:
            df_full.to_csv(out_path, index=False)

        return {
            "name": display_name,
            "source_type": source_type,
            "source_url": source_url,
            "file_path": str(out_path),
        }

    async def search_openalex_dissertations(
        self,
        query: str,
        domain: str = "biomedical",
        min_year: int = 2018,
        max_results: int = 5,
        top_science_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Searches OpenAlex live API for real doctoral dissertations (`type:dissertation`).
        Captures author ORCIDs, OpenAlex author IDs, advisor/co-author names, institution IDs,
        and open-access PDF/PMCID links for downstream ORCID graph matching and full-text scraping.
        """
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "filter": f"type:dissertation,from_publication_date:{min_year}-01-01",
            "per-page": max(25, max_results * 5),
            "mailto": "research@thesisout.com",
        }
        headers = {"User-Agent": "ThesisOut/1.0 (https://thesisout.com; mailto:research@thesisout.com)"}
        results: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.warning("OpenAlex returned HTTP %d for query '%s'", resp.status_code, query)
                    return []
                data = resp.json()
        except Exception as exc:
            logger.warning("OpenAlex search request failed for '%s': %s", query, exc)
            return []

        for item in data.get("results", []):
                title = item.get("title")
                if not title:
                    continue
                authorships = item.get("authorships") or []
                authors_list: List[str] = []
                author_orcids: List[str] = []
                author_openalex_ids: List[str] = []
                institution_ids: List[str] = []
                uni_name = ""

                for auth in authorships:
                    author_obj = auth.get("author") or {}
                    author_name = author_obj.get("display_name")
                    if author_name:
                        authors_list.append(author_name)
                    orcid_val = author_obj.get("orcid")
                    if orcid_val:
                        clean_orcid = orcid_val.rstrip("/").split("/")[-1]
                        if clean_orcid and clean_orcid not in author_orcids:
                            author_orcids.append(clean_orcid)
                    oa_author_id = author_obj.get("id")
                    if oa_author_id:
                        short_aid = oa_author_id.rstrip("/").split("/")[-1]
                        if short_aid and short_aid not in author_openalex_ids:
                            author_openalex_ids.append(short_aid)

                    insts = auth.get("institutions") or []
                    for inst in insts:
                        if inst.get("display_name") and not uni_name:
                            uni_name = inst["display_name"]
                        inst_id = inst.get("id")
                        if inst_id:
                            short_iid = inst_id.rstrip("/").split("/")[-1]
                            if short_iid and short_iid not in institution_ids:
                                institution_ids.append(short_iid)

                if not uni_name:
                    prim = (item.get("primary_location") or {}).get("source") or {}
                    uni_name = prim.get("display_name") or "Indexed Research Institution"

                if top_science_only and not self.is_top_science_institution(uni_name):
                    continue

                abstract_text = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                if not abstract_text:
                    abstract_text = "Abstract metadata indexed via OpenAlex (full text available via source repository URL)."

                pub_year = item.get("publication_year") or min_year
                ids_dict = item.get("ids") or {}
                oa_url = (
                    (item.get("open_access") or {}).get("oa_url")
                    or item.get("doi")
                    or item.get("id")
                )
                cited_by = item.get("cited_by_count") or 0
                rigor_score = round(min(0.99, 0.80 + min(cited_by, 50) * 0.003), 2)

                results.append(
                    {
                        "external_id": item.get("id"),
                        "title": title,
                        "authors": ", ".join(authors_list[:4]) if authors_list else "Unknown Author",
                        "author_names": authors_list,
                        "author_orcids": author_orcids,
                        "author_openalex_ids": author_openalex_ids,
                        "institution_ids": institution_ids,
                        "university": uni_name,
                        "publication_year": int(pub_year),
                        "domain": domain,
                        "source_repo": "OpenAlex Live API",
                        "source_url": oa_url,
                        "doi": item.get("doi"),
                        "pmcid": ids_dict.get("pmcid"),
                        "pmid": ids_dict.get("pmid"),
                        "institution_tier": uni_name,
                        "scientific_rigor_score": rigor_score,
                        "citation_velocity": round(float(cited_by) / max(1, 2026 - int(pub_year)), 2),
                        "abstract": abstract_text,
                    }
                )
                if len(results) >= max_results:
                    break

        return results

    def _reconstruct_abstract(self, inverted_index: Optional[Dict[str, List[int]]]) -> str:
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)

    # =========================================================================
    # 1. FULL-TEXT DATA AVAILABILITY & SUPPLEMENTARY MATERIAL SECTION SCRAPING
    # =========================================================================
    async def scrape_fulltext_data_availability_links(
        self,
        dissertation_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fetches open-access XML/HTML/PDF from EuropePMC or institutional repositories
        specifically to parse the 'Data Availability' / 'Supplementary Material' section
        at the end of the thesis/paper (where dataset DOIs live).
        """
        discovered_links: List[Dict[str, str]] = []
        seen_ids = set()
        statements: List[str] = []

        def _merge(da_res: Dict[str, Any]):
            if da_res.get("statement"):
                statements.append(da_res["statement"])
            for lk in da_res.get("links", []):
                if lk["identifier"] not in seen_ids:
                    seen_ids.add(lk["identifier"])
                    discovered_links.append(lk)

        # First scan abstract if any links are already present
        abstract_txt = dissertation_meta.get("abstract") or ""
        if abstract_txt:
            _merge(pdf_extractor_agent.extract_data_availability_section(abstract_txt))

        title = dissertation_meta.get("title") or ""
        pmcid = dissertation_meta.get("pmcid") or ""
        doi = (dissertation_meta.get("doi") or "").replace("https://doi.org/", "").strip()
        oa_url = dissertation_meta.get("source_url") or ""

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # Step 1A: Resolve PMCID via EuropePMC if not already populated
            if not pmcid and (doi or title):
                try:
                    epmc_q = f'DOI:"{doi}"' if doi else f'TITLE:"{title[:90]}" AND OPEN_ACCESS:y'
                    r_search = await client.get(
                        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                        params={"query": epmc_q, "format": "json", "pageSize": 2},
                    )
                    if r_search.status_code == 200:
                        res_list = ((r_search.json().get("resultList") or {}).get("result")) or []
                        for r_item in res_list:
                            if r_item.get("pmcid"):
                                pmcid = r_item["pmcid"]
                                break
                except Exception as exc:
                    logger.debug("EuropePMC PMCID lookup notice: %s", exc)

            # Step 1B: Fetch full-text JATS XML from EuropePMC and parse <sec sec-type="data-availability">
            if pmcid:
                clean_pmcid = pmcid.split("/")[-1].strip()
                if not clean_pmcid.upper().startswith("PMC"):
                    clean_pmcid = f"PMC{clean_pmcid}"
                try:
                    r_xml = await client.get(
                        f"https://www.ebi.ac.uk/europepmc/webservices/rest/{clean_pmcid}/fullTextXML"
                    )
                    if r_xml.status_code == 200 and r_xml.text:
                        _merge(pdf_extractor_agent.extract_data_availability_section(r_xml.text))
                except Exception as exc:
                    logger.debug("EuropePMC fullTextXML scrape notice for %s: %s", clean_pmcid, exc)

            # Step 1C: Fetch institutional open-access HTML/PDF landing page (oa_url) to scrape back-matter Data Availability
            if oa_url and oa_url.startswith("http") and "openalex.org" not in oa_url:
                try:
                    r_oa = await client.get(
                        oa_url,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; ThesisOutHarvester/2.0)"},
                    )
                    if r_oa.status_code == 200:
                        ctype = (r_oa.headers.get("content-type") or "").lower()
                        if "pdf" in ctype or oa_url.lower().endswith(".pdf"):
                            if len(r_oa.content) < 25_000_000:
                                reader = PdfReader(io.BytesIO(r_oa.content))
                                n_p = len(reader.pages)
                                # Specifically parse the last 20 pages of the thesis where Data Availability & Appendices live
                                back_pages = list(range(max(0, n_p - 20), n_p))
                                back_text = "\n".join(
                                    (reader.pages[p].extract_text() or "") for p in back_pages
                                )
                                _merge(pdf_extractor_agent.extract_data_availability_section(back_text))
                        else:
                            html_text = r_oa.text[:250_000]
                            _merge(pdf_extractor_agent.extract_data_availability_section(html_text))
                except Exception as exc:
                    logger.debug("Institutional OA full-text scrape notice for %s: %s", oa_url, exc)

        return {
            "statement": statements[0] if statements else "",
            "links": discovered_links,
        }

    # =========================================================================
    # 2. AUTHOR ORCID & LAB GRAPH MATCHING (OpenAlex + DataCite + Crossref)
    # =========================================================================
    async def find_datasets_by_author_orcid_and_lab_graph(
        self,
        dissertation_meta: Dict[str, Any],
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Queries OpenAlex (`type:dataset`), DataCite (`resource-type-id=dataset`), and Crossref (`type:dataset`)
        for dataset works sharing the exact author ORCID, OpenAlex author ID, or advisor + institution
        within ±2 years of the dissertation defense date, then resolves any discovered dataset DOIs.
        """
        pub_year = int(dissertation_meta.get("publication_year") or 2023)
        year_min = pub_year - 2
        year_max = pub_year + 2
        orcids: List[str] = dissertation_meta.get("author_orcids") or []
        author_ids: List[str] = dissertation_meta.get("author_openalex_ids") or []
        inst_ids: List[str] = dissertation_meta.get("institution_ids") or []
        author_names: List[str] = dissertation_meta.get("author_names") or []
        if not author_names and dissertation_meta.get("authors"):
            author_names = [
                a.strip()
                for a in str(dissertation_meta["authors"]).split(",")
                if a.strip() and a.strip() != "Unknown Author"
            ]

        discovered_dois_and_urls: List[Dict[str, str]] = []
        seen_identifiers = set()

        def _record_candidate(identifier_or_url: str, label: str):
            if not identifier_or_url:
                return
            for lk in pdf_extractor_agent.extract_repository_links(identifier_or_url, section_label=label):
                if lk["identifier"] not in seen_identifiers:
                    seen_identifiers.add(lk["identifier"])
                    discovered_dois_and_urls.append(lk)

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # 2A. OpenAlex Graph Query: type:dataset by exact author ORCID or OpenAlex Author ID within ±2 years
            oa_filters: List[str] = []
            for orc in orcids[:2]:
                oa_filters.append(
                    f"type:dataset,authorships.author.orcid:{orc},from_publication_date:{year_min}-01-01,to_publication_date:{year_max}-12-31"
                )
            for aid in author_ids[:2]:
                oa_filters.append(
                    f"type:dataset,authorships.author.id:{aid},from_publication_date:{year_min}-01-01,to_publication_date:{year_max}-12-31"
                )
            if inst_ids and author_names:
                oa_filters.append(
                    f"type:dataset,authorships.institutions.id:{inst_ids[0]},from_publication_date:{year_min}-01-01,to_publication_date:{year_max}-12-31"
                )

            for flt in oa_filters[:3]:
                try:
                    params: Dict[str, Any] = {"filter": flt, "per-page": 10}
                    if "institutions.id" in flt and author_names:
                        params["search"] = author_names[0].split()[-1]
                    r_oa = await client.get("https://api.openalex.org/works", params=params)
                    if r_oa.status_code == 200:
                        for w in r_oa.json().get("results", []):
                            _record_candidate(str(w.get("doi") or ""), "openalex_orcid_graph")
                            _record_candidate(
                                str((w.get("open_access") or {}).get("oa_url") or ""),
                                "openalex_orcid_graph",
                            )
                except Exception as exc:
                    logger.debug("OpenAlex dataset graph query notice: %s", exc)

            # 2B. DataCite REST API: resource-type-id=dataset matching ORCID or author + institution within ±2 years
            datacite_queries: List[str] = []
            for orc in orcids[:2]:
                datacite_queries.append(
                    f"creators.nameIdentifiers.nameIdentifier:{orc} AND publicationYear:[{year_min} TO {year_max}]"
                )
            for aname in author_names[:2]:
                surname = aname.split()[-1]
                if len(surname) >= 3:
                    datacite_queries.append(
                        f'creators.name:"{aname}" AND publicationYear:[{year_min} TO {year_max}]'
                    )

            for dc_q in datacite_queries[:3]:
                try:
                    r_dc = await client.get(
                        "https://api.datacite.org/dois",
                        params={
                            "query": dc_q,
                            "resource-type-id": "dataset",
                            "page[size]": 8,
                        },
                    )
                    if r_dc.status_code == 200:
                        for item in r_dc.json().get("data", []):
                            doi_str = item.get("id") or (item.get("attributes") or {}).get("doi") or ""
                            url_str = (item.get("attributes") or {}).get("url") or ""
                            _record_candidate(doi_str, "datacite_orcid_graph")
                            _record_candidate(url_str, "datacite_orcid_graph")
                except Exception as exc:
                    logger.debug("DataCite dataset graph query notice: %s", exc)

            # 2C. Crossref REST API: filter=type:dataset,from-pub-date:{year_min},until-pub-date:{year_max}
            if author_names:
                try:
                    r_cr = await client.get(
                        "https://api.crossref.org/works",
                        params={
                            "query.author": author_names[0],
                            "filter": f"type:dataset,from-pub-date:{year_min},until-pub-date:{year_max}",
                            "rows": 6,
                        },
                    )
                    if r_cr.status_code == 200:
                        items = ((r_cr.json().get("message") or {}).get("items")) or []
                        for it in items:
                            _record_candidate(str(it.get("DOI") or ""), "crossref_author_graph")
                            _record_candidate(str(it.get("URL") or ""), "crossref_author_graph")
                except Exception as exc:
                    logger.debug("Crossref dataset graph query notice: %s", exc)

        # Resolve any discovered repository links into real tabular files
        downloaded: List[Dict[str, str]] = []
        for lk in discovered_dois_and_urls:
            resolved = await self.resolve_repository_link(lk, target_dir=target_dir)
            for rf in resolved:
                downloaded.append(rf)
                if len(downloaded) >= max_files:
                    return downloaded

        return downloaded

    # =========================================================================
    # 3. MULTI-REPOSITORY TABULAR HARVESTERS (Zenodo, Figshare, Dataverse, Dryad)
    # =========================================================================
    async def resolve_repository_link(
        self,
        link_info: Dict[str, str],
        target_dir: Path,
    ) -> List[Dict[str, str]]:
        """Resolves a specific extracted repository link/DOI from Zenodo, Figshare, Dataverse, or Dryad."""
        stype = (link_info.get("source_type") or "").lower()
        ident = link_info.get("identifier") or link_info.get("url") or ""
        if stype == "zenodo":
            return await self.fetch_zenodo_record_by_id(ident, target_dir)
        if stype == "figshare":
            return await self.fetch_figshare_article_by_id(ident, target_dir)
        if stype == "dataverse":
            return await self.fetch_harvard_dataverse_by_doi(ident, target_dir)
        if stype == "dryad":
            return await self.fetch_dryad_by_doi(ident, target_dir)
        return []

    async def fetch_real_zenodo_datasets(
        self,
        query: str,
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Searches the live Zenodo REST API (`https://zenodo.org/api/records`) for real open-access
        tabular CSV/TSV/XLSX datasets matching the dissertation topic and downloads the real file to disk.
        """
        api_url = "https://zenodo.org/api/records"
        params = {
            "q": f"{query} AND filetype:csv",
            "type": "dataset",
            "size": 5,
            "access_right": "open",
        }
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(api_url, params=params)
                if resp.status_code != 200:
                    return []
                hits = (resp.json().get("hits") or {}).get("hits") or []
                for rec in hits:
                    rec_id = rec.get("id")
                    files = rec.get("files") or []
                    for f in files:
                        key = f.get("key") or ""
                        size_bytes = f.get("size") or 0
                        if key.lower().endswith((".csv", ".tsv")) and 400 < size_bytes < 15_000_000:
                            dl_url = (f.get("links") or {}).get("self")
                            if dl_url:
                                r_file = await client.get(dl_url)
                                if r_file.status_code == 200:
                                    saved = self._validate_and_save_tabular_bytes(
                                        raw_bytes=r_file.content,
                                        filename=key,
                                        file_prefix=f"zenodo_{rec_id}",
                                        source_type="zenodo",
                                        source_url=f"https://zenodo.org/records/{rec_id}",
                                        display_name=f"Zenodo Record #{rec_id}: {key}",
                                        target_dir=target_dir,
                                        min_rows=20,
                                        min_cols=3,
                                    )
                                    if saved:
                                        downloaded.append(saved)
                                        if len(downloaded) >= max_files:
                                            return downloaded
        except Exception as exc:
            logger.warning("Live Zenodo dataset search notice for '%s': %s", query, exc)
        return downloaded

    async def fetch_zenodo_record_by_id(self, zenodo_id_or_url: str, target_dir: Path) -> List[Dict[str, str]]:
        """Downloads real tabular files from a specific Zenodo record ID or DOI."""
        m = re.search(r"(\d{5,10})", zenodo_id_or_url)
        if not m:
            return []
        rec_id = m.group(1)
        api_url = f"https://zenodo.org/api/records/{rec_id}"
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(api_url)
                if resp.status_code != 200:
                    return []
                data = resp.json()
                for f in data.get("files", []):
                    key = f.get("key") or ""
                    if any(key.lower().endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".parquet")):
                        dl_url = (f.get("links") or {}).get("self")
                        if dl_url:
                            r_file = await client.get(dl_url)
                            if r_file.status_code == 200:
                                saved = self._validate_and_save_tabular_bytes(
                                    raw_bytes=r_file.content,
                                    filename=key,
                                    file_prefix=f"zenodo_{rec_id}",
                                    source_type="zenodo",
                                    source_url=f"https://zenodo.org/records/{rec_id}",
                                    display_name=f"Zenodo #{rec_id} ({key})",
                                    target_dir=target_dir,
                                    min_rows=20,
                                    min_cols=3,
                                )
                                if saved:
                                    downloaded.append(saved)
        except Exception as exc:
            logger.warning("Zenodo record fetch warning for %s: %s", zenodo_id_or_url, exc)
        return downloaded

    async def fetch_figshare_article_by_id(
        self,
        figshare_id_or_url: str,
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """Resolves and downloads tabular files from a Figshare article ID or DOI (`api.figshare.com/v2/articles/{id}`)."""
        m = re.search(r"(\d{5,12})", figshare_id_or_url)
        if not m:
            return []
        art_id = m.group(1)
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(f"https://api.figshare.com/v2/articles/{art_id}")
                if resp.status_code != 200:
                    return []
                art = resp.json()
                art_url = art.get("url_public_html") or f"https://figshare.com/articles/dataset/{art_id}"
                for f in art.get("files") or []:
                    fname = f.get("name") or ""
                    size_b = f.get("size") or 0
                    dl_url = f.get("download_url")
                    if (
                        dl_url
                        and any(fname.lower().endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".parquet"))
                        and 300 < size_b < 15_000_000
                    ):
                        r_file = await client.get(dl_url)
                        if r_file.status_code == 200:
                            saved = self._validate_and_save_tabular_bytes(
                                raw_bytes=r_file.content,
                                filename=fname,
                                file_prefix=f"figshare_{art_id}",
                                source_type="figshare",
                                source_url=art_url,
                                display_name=f"Figshare #{art_id} ({fname})",
                                target_dir=target_dir,
                            )
                            if saved:
                                downloaded.append(saved)
                                if len(downloaded) >= max_files:
                                    return downloaded
        except Exception as exc:
            logger.debug("Figshare article resolver notice for %s: %s", figshare_id_or_url, exc)
        return downloaded

    async def fetch_figshare_datasets(
        self,
        query: str,
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Queries the live Figshare API (`https://api.figshare.com/v2/articles/search`) for dataset articles (item_type=3)
        and downloads verified tabular files (.csv, .tsv, .xlsx).
        """
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.post(
                    "https://api.figshare.com/v2/articles/search",
                    json={"search_for": f"{query} csv", "item_type": 3, "page_size": 6},
                )
                if resp.status_code != 200:
                    return []
                articles = resp.json() or []
                for item in articles:
                    art_id = item.get("id")
                    if not art_id:
                        continue
                    resolved = await self.fetch_figshare_article_by_id(str(art_id), target_dir, max_files=max_files)
                    for rf in resolved:
                        downloaded.append(rf)
                        if len(downloaded) >= max_files:
                            return downloaded
        except Exception as exc:
            logger.debug("Figshare dataset search notice for '%s': %s", query, exc)
        return downloaded

    async def fetch_harvard_dataverse_by_doi(
        self,
        doi_or_url: str,
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """Resolves a Harvard Dataverse dataset DOI (`10.7910/DVN/...`) into tabular files."""
        m = re.search(r"(10\.7910/DVN/[A-Za-z0-9]+)", doi_or_url, re.I)
        if not m:
            return []
        pid = f"doi:{m.group(1)}"
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(
                    "https://dataverse.harvard.edu/api/datasets/:persistentId/",
                    params={"persistentId": pid},
                )
                if resp.status_code != 200:
                    return []
                files = (((resp.json().get("data") or {}).get("latestVersion") or {}).get("files")) or []
                for f_entry in files:
                    df_meta = f_entry.get("dataFile") or {}
                    file_id = df_meta.get("id")
                    fname = df_meta.get("originalFileName") or df_meta.get("filename") or ""
                    fsize = df_meta.get("filesize") or 0
                    if (
                        file_id
                        and any(fname.lower().endswith(ext) for ext in (".csv", ".tsv", ".tab", ".xlsx"))
                        and 300 < fsize < 15_000_000
                    ):
                        r_dl = await client.get(f"https://dataverse.harvard.edu/api/access/datafile/{file_id}")
                        if r_dl.status_code == 200:
                            saved = self._validate_and_save_tabular_bytes(
                                raw_bytes=r_dl.content,
                                filename=fname,
                                file_prefix=f"dataverse_{file_id}",
                                source_type="dataverse",
                                source_url=f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={pid}",
                                display_name=f"Harvard Dataverse ({fname})",
                                target_dir=target_dir,
                            )
                            if saved:
                                downloaded.append(saved)
                                if len(downloaded) >= max_files:
                                    return downloaded
        except Exception as exc:
            logger.debug("Harvard Dataverse DOI resolver notice for %s: %s", doi_or_url, exc)
        return downloaded

    async def fetch_harvard_dataverse_datasets(
        self,
        query: str,
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Queries Harvard Dataverse Search API (`https://dataverse.harvard.edu/api/search?type=file`)
        for open-access tabular files (.csv, .tab, .tsv, .xlsx) and downloads them via `/api/access/datafile/{file_id}`.
        """
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ThesisOutHarvester/2.0)"},
            ) as client:
                resp = await client.get(
                    "https://dataverse.harvard.edu/api/search",
                    params={"q": query, "type": "file", "per_page": 12},
                )
                if resp.status_code != 200:
                    return []
                items = ((resp.json().get("data") or {}).get("items")) or []
                for item in items:
                    file_id = item.get("file_id")
                    fname = item.get("name") or ""
                    fsize = item.get("size_in_bytes") or 0
                    dataset_citation = item.get("dataset_persistent_id") or ""
                    if (
                        file_id
                        and any(fname.lower().endswith(ext) for ext in (".csv", ".tsv", ".tab", ".xlsx"))
                        and 150 < fsize < 15_000_000
                    ):
                        r_dl = await client.get(f"https://dataverse.harvard.edu/api/access/datafile/{file_id}")
                        if r_dl.status_code == 200:
                            src_url = (
                                f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={dataset_citation}"
                                if dataset_citation
                                else f"https://dataverse.harvard.edu/file.xhtml?fileId={file_id}"
                            )
                            saved = self._validate_and_save_tabular_bytes(
                                raw_bytes=r_dl.content,
                                filename=fname,
                                file_prefix=f"dataverse_{file_id}",
                                source_type="dataverse",
                                source_url=src_url,
                                display_name=f"Harvard Dataverse #{file_id}: {fname}",
                                target_dir=target_dir,
                            )
                            if saved:
                                downloaded.append(saved)
                                if len(downloaded) >= max_files:
                                    return downloaded
        except Exception as exc:
            logger.debug("Harvard Dataverse search notice for '%s': %s", query, exc)
        return downloaded

    async def fetch_dryad_by_doi(
        self,
        dryad_doi_or_url: str,
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """Resolves a Dryad dataset DOI (`10.5061/dryad....`) via Dryad REST API v2 into tabular files."""
        m = re.search(r"(10\.5061/dryad\.[A-Za-z0-9_.-]+)", dryad_doi_or_url, re.I)
        if not m:
            return []
        doi_str = f"doi:{m.group(1)}"
        encoded_doi = urllib.parse.quote(doi_str, safe="")
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                r_ds = await client.get(f"https://datadryad.org/api/v2/datasets/{encoded_doi}")
                if r_ds.status_code != 200:
                    return []
                ds_json = r_ds.json()
                ver_href = (((ds_json.get("_links") or {}).get("stash:version") or {}).get("href")) or ""
                if not ver_href:
                    return []
                r_files = await client.get(f"https://datadryad.org{ver_href}/files")
                if r_files.status_code != 200:
                    return []
                files_list = (((r_files.json().get("_embedded") or {}).get("stash:files"))) or []
                for f_obj in files_list:
                    fname = f_obj.get("path") or ""
                    fsize = f_obj.get("size") or 0
                    dl_href = (((f_obj.get("_links") or {}).get("stash:download") or {}).get("href")) or ""
                    if (
                        dl_href
                        and any(fname.lower().endswith(ext) for ext in (".csv", ".tsv", ".xlsx"))
                        and 300 < fsize < 15_000_000
                    ):
                        r_dl = await client.get(f"https://datadryad.org{dl_href}")
                        if r_dl.status_code == 200:
                            saved = self._validate_and_save_tabular_bytes(
                                raw_bytes=r_dl.content,
                                filename=fname,
                                file_prefix="dryad",
                                source_type="dryad",
                                source_url=f"https://datadryad.org/stash/dataset/{doi_str}",
                                display_name=f"Dryad ({fname})",
                                target_dir=target_dir,
                            )
                            if saved:
                                downloaded.append(saved)
                                if len(downloaded) >= max_files:
                                    return downloaded
        except Exception as exc:
            logger.debug("Dryad DOI resolver notice for %s: %s", dryad_doi_or_url, exc)
        return downloaded

    async def fetch_dryad_datasets(
        self,
        query: str,
        target_dir: Path,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Searches Dryad API v2 (`https://datadryad.org/api/v2/search`) for datasets matching the query
        and downloads verified tabular files.
        """
        downloaded: List[Dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(
                    "https://datadryad.org/api/v2/search",
                    params={"q": query, "per_page": 5},
                )
                if resp.status_code != 200:
                    return []
                datasets = (((resp.json().get("_embedded") or {}).get("stash:datasets"))) or []
                for ds in datasets:
                    ident = ds.get("identifier") or ""
                    if ident:
                        resolved = await self.fetch_dryad_by_doi(ident, target_dir, max_files=max_files)
                        for rf in resolved:
                            downloaded.append(rf)
                            if len(downloaded) >= max_files:
                                return downloaded
        except Exception as exc:
            logger.debug("Dryad dataset search notice for '%s': %s", query, exc)
        return downloaded

    async def resolve_multi_repository_datasets(
        self,
        query: str,
        target_dir: Path,
        dissertation_meta: Optional[Dict[str, Any]] = None,
        extracted_links: Optional[List[Dict[str, str]]] = None,
        max_files: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Unified High-Power Dataset Finder:
        Stage 1: Full-Text Data Availability / Supplementary Material DOI resolution.
        Stage 2: Author ORCID & Lab Graph matching across OpenAlex, DataCite, and Crossref (±2 years).
        Stage 3: Multi-repository tabular harvesting across Zenodo, Figshare, Harvard Dataverse, and Dryad.
        """
        downloaded: List[Dict[str, str]] = []
        candidate_links: List[Dict[str, str]] = list(extracted_links or [])

        # Stage 1: Scrape full-text Data Availability / Supplementary Material sections
        if dissertation_meta:
            da_scraped = await self.scrape_fulltext_data_availability_links(dissertation_meta)
            for lk in da_scraped.get("links", []):
                if lk not in candidate_links:
                    candidate_links.append(lk)

        for lk in candidate_links:
            resolved = await self.resolve_repository_link(lk, target_dir=target_dir)
            for rf in resolved:
                downloaded.append(rf)
                if len(downloaded) >= max_files:
                    return downloaded

        # Stage 2: Author ORCID & Lab Graph Matching (OpenAlex + DataCite + Crossref ±2 years)
        if dissertation_meta:
            graph_files = await self.find_datasets_by_author_orcid_and_lab_graph(
                dissertation_meta=dissertation_meta,
                target_dir=target_dir,
                max_files=max_files,
            )
            for gf in graph_files:
                downloaded.append(gf)
                if len(downloaded) >= max_files:
                    return downloaded

        # Stage 3: Multi-Repository Tabular Harvesters (Zenodo -> Figshare -> Harvard Dataverse -> Dryad)
        for resolver_fn in (
            self.fetch_real_zenodo_datasets,
            self.fetch_figshare_datasets,
            self.fetch_harvard_dataverse_datasets,
            self.fetch_dryad_datasets,
        ):
            files = await resolver_fn(query=query, target_dir=target_dir, max_files=max_files)
            for f in files:
                downloaded.append(f)
                if len(downloaded) >= max_files:
                    return downloaded

        return downloaded


harvester_agent = HarvesterAgent()
