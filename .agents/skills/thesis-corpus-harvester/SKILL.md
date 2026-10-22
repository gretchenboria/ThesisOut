---
name: thesis-corpus-harvester
description: Use this skill when harvesting doctoral dissertations from OpenAlex, EuropePMC, or arXiv and resolving open-access tabular datasets (.csv, .tsv, .xlsx, .parquet) from Zenodo REST API.
---

# Thesis Corpus & Open-Access Dataset Harvester Skill

## Overview
Queries live academic archives (`https://api.openalex.org/works`, `https://www.ebi.ac.uk/europepmc/webservices/rest/search`, `http://export.arxiv.org/api/query`) and pairs discovered studies with verified open-access tabular datasets from Zenodo (`https://zenodo.org/api/records`).

## Execution Steps
1. Query OpenAlex with `filter=type:dissertation,from_publication_date:2018-01-01` and reconstruct the inverted abstract index.
2. Query Zenodo REST API (`https://zenodo.org/api/records`) with `type=dataset&access_right=open` or fetch specific record IDs via `/api/records/{record_id}`.
3. Validate downloaded CSV/TSV files using `data_scientist_agent.load_dataframe`:
   - Must contain `>= 20` rows and `>= 3` non-identifier columns.
   - Must not contain synthetic or random fallback data.
4. Persist `Dissertation` and `DatasetArtifact` rows in SQLite with full `external_id` and `source_url` provenance.
