---
name: thesis-corpus-harvester
description: Use this skill when harvesting doctoral dissertations from OpenAlex, EuropePMC, or arXiv and resolving open-access tabular datasets (.csv, .tsv, .tab, .xlsx, .parquet) via 3-stage dataset discovery (Full-Text Data Availability Scraping, Author ORCID & Lab Graph Matching across OpenAlex/DataCite/Crossref ±2 years, and Multi-Repository Tabular Harvesters across Zenodo, Figshare, Harvard Dataverse, and Dryad).
---

# Thesis Corpus & Multi-Repository Dataset Finder Skill

## Overview
Queries live academic archives (`OpenAlex`, `EuropePMC`, `arXiv`) and resolves empirical tabular datasets via three high-recall discovery stages:
1. **Full-Text Data Availability & Supplementary Material Scraping**: Fetches open-access JATS XML from EuropePMC (`/fullTextXML`) and back-matter pages of institutional open-access dissertation PDFs to extract embedded repository DOIs (`10.5281/zenodo.*`, `10.6084/m9.figshare.*`, `10.7910/DVN/*`, `10.5061/dryad.*`).
2. **Author ORCID & Lab Graph Matching (`OpenAlex` + `DataCite` + `Crossref`)**: Queries `OpenAlex` (`type:dataset`), `DataCite` (`resource-type-id=dataset`), and `Crossref` (`type:dataset`) for dataset works sharing the exact author ORCID, OpenAlex author ID, or advisor + institution within `±2` years of the dissertation defense date.
3. **Multi-Repository Tabular Harvesters**: Direct file-download resolvers for **Zenodo** (`zenodo.org/api/records`), **Figshare** (`api.figshare.com/v2/articles`), **Harvard Dataverse** (`dataverse.harvard.edu/api/search?type=file`), and **Dryad** (`datadryad.org/api/v2/search`).

## Execution Steps
1. Query OpenAlex with `filter=type:dissertation` and extract `author_orcids`, `author_openalex_ids`, `institution_ids`, `doi`, `pmcid`, and `oa_url`.
2. Run `harvester_agent.resolve_multi_repository_datasets(...)` to execute Stage 1 (Full-Text Data Availability scraping), Stage 2 (ORCID & Lab Graph matching ±2 years), and Stage 3 (Zenodo, Figshare, Harvard Dataverse, and Dryad tabular downloads).
3. Validate downloaded files and persist `Dissertation` and `DatasetArtifact` records in SQLite with full provenance.
