# Graph Report - Pharmint  (2026-09-22)

## Corpus Check
- 31 files · ~42,102 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 208 nodes · 454 edges · 17 communities (9 shown, 8 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a7082825`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- model_ab.py
- mixed_ac.py
- Model D — Arellano–Bond Difference GMM, Corrected
- Code.py
- Backtesting & Stability Reports
- ETL Pipeline Reference & Run Log
- Graphify Skill Documentation
- model_seasonal.py
- LSTMRegressor
- Mixed Model Results Docs
- Pipeline Run Log
- Dataset Merge Script
- Drug Substance Splitter
- Monthly Transactions Chart
- Model Ensemble Results (Drug Grain)
- Model Ensemble Results (Country Grain)
- PharmInt ETL Pipeline Reference Page

## God Nodes (most connected - your core abstractions)
1. `_train_window()` - 25 edges
2. `main()` - 20 edges
3. `score_counts()` - 18 edges
4. `compute_entity_effects()` - 18 edges
5. `main()` - 16 edges
6. `load_panel()` - 16 edges
7. `split_substances()` - 16 edges
8. `oos_r2()` - 16 edges
9. `main()` - 15 edges
10. `fit_model_d_ab_gmm()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `country_market_intel (Code.py job output)` --semantically_similar_to--> `country_market_intel (Mongo aggregation collection)`  [INFERRED] [semantically similar]
  Code results/pipeline_run_log.txt → etl-pipeline.html
- `exporter_directory (Code.py job output)` --semantically_similar_to--> `exporter_directory (Mongo aggregation collection)`  [INFERRED] [semantically similar]
  Code results/pipeline_run_log.txt → etl-pipeline.html
- `market_opportunities (Code.py job output)` --semantically_similar_to--> `market_opportunities (Mongo aggregation collection)`  [INFERRED] [semantically similar]
  Code results/pipeline_run_log.txt → etl-pipeline.html
- `price_trends_monthly (Code.py job output)` --semantically_similar_to--> `price_trends_monthly (Mongo aggregation collection)`  [INFERRED] [semantically similar]
  Code results/pipeline_run_log.txt → etl-pipeline.html
- `product_analytics (Code.py job output)` --semantically_similar_to--> `product_analytics (Mongo aggregation collection)`  [INFERRED] [semantically similar]
  Code results/pipeline_run_log.txt → etl-pipeline.html

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Drug×Country Grain Fallback Logic (Models A/B/D)** — models_two_model_spec_model_a, models_two_model_spec_model_b, models_two_model_spec_model_d, models_two_model_spec_country_grain [EXTRACTED 0.90]
- **Shared Quality-Filtered Source Feeding Six Aggregation Jobs** — etl_pipeline_qualityfilters, etl_pipeline_product_analytics, etl_pipeline_country_market_intel, etl_pipeline_exporter_directory, etl_pipeline_price_trends_monthly, etl_pipeline_market_opportunities, etl_pipeline_regional_analytics [EXTRACTED 1.00]
- **graphify Skill Documentation Set** — _claude_skills_graphify_skill_pipeline, _claude_skills_graphify_references_add_watch_guide, _claude_skills_graphify_references_exports_guide, _claude_skills_graphify_references_extraction_spec_guide, _claude_skills_graphify_references_github_and_merge_guide, _claude_skills_graphify_references_hooks_guide, _claude_skills_graphify_references_query_guide, _claude_skills_graphify_references_transcribe_guide, _claude_skills_graphify_references_update_guide [EXTRACTED 1.00]
- **Mixed Model Comparison Experiment (PPML as common baseline)** — code_results_mixed_ac_results_results_doc, code_results_mixed_dc_results_results_doc, code_results_mixed_ac_results_model_c_ppml [INFERRED 0.85]

## Communities (17 total, 8 thin omitted)

### Community 0 - "model_ab.py"
Cohesion: 0.13
Nodes (41): adj_r2(), ar_diagnostics(), compute_entity_scale_ppml(), fit_model_a(), fit_model_b_gmm(), fit_model_b_ols_within(), fit_model_d_ab_gmm(), fit_pooled_ols() (+33 more)

### Community 1 - "mixed_ac.py"
Cohesion: 0.11
Nodes (30): _load_base(), main(), _metric_map(), Shared helper for folding a satellite model's test-set scores (Mixed (A/C),…, Rebuild model_comparision.xlsx from model_comparision.csv alone (no satellite…, metric label (as it appears in model_comparision's "metric" column) -> key in…, Load model_comparision.xlsx if it exists (stripping its NOTE row so it can be…, Update model_comparision.xlsx in place with one satellite model's scores,… (+22 more)

### Community 2 - "Model D — Arellano–Bond Difference GMM, Corrected"
Cohesion: 0.18
Nodes (17): Model GBM Results, Model LSTM Results, Model Results A-D (Drug Grain), Model Results A-D (Country Grain), Two-Model Cheatsheet, Two-Model Working Spec, Arellano–Bond Difference GMM Method, Bracket Check Diagnostic (+9 more)

### Community 3 - "Code.py"
Cohesion: 0.12
Nodes (31): apply_quality_filters(), clean_product_key(), enrich(), job_country_market_intel(), job_exporter_directory(), job_market_opportunities(), job_price_trends_monthly(), job_product_analytics() (+23 more)

### Community 4 - "Backtesting & Stability Reports"
Cohesion: 0.21
Nodes (14): _apply_color_scale(), canonical_name(), main(), _model_sort_key(), parse_rows(), _pivot_metric(), Backtests model_ab.py (Models A/B/C/D) and model_ensemble.py (Models F/G) at…, One row per model, one column per training window, plus stability columns. CV%… (+6 more)

### Community 5 - "ETL Pipeline Reference & Run Log"
Cohesion: 0.14
Nodes (14): country_market_intel (Code.py job output), exporter_directory (Code.py job output), market_opportunities (Code.py job output), price_trends_monthly (Code.py job output), product_analytics (Code.py job output), regional_analytics (Code.py job output), country_market_intel (Mongo aggregation collection), exporter_directory (Mongo aggregation collection) (+6 more)

### Community 6 - "Graphify Skill Documentation"
Cohesion: 0.20
Nodes (11): .claude/CLAUDE.md graphify Pointer, add-watch.md Reference, exports.md Reference, extraction-spec.md Reference, github-and-merge.md Reference, hooks.md Reference, query.md Reference, transcribe.md Reference (+3 more)

### Community 7 - "model_seasonal.py"
Cohesion: 0.21
Nodes (16): compute_entity_effects(), predict_and_score(), Plug-in substance fixed effect: alpha_d = mean over the substance's own…, add_fourier(), append_to_comparison_csv(), fit_model_h_fourier_gmm(), fit_sarima_aggregate(), fit_sarima_per_substance() (+8 more)

### Community 9 - "Mixed Model Results Docs"
Cohesion: 0.57
Nodes (7): Mixed A/C Per-Substance Routing, Model A (AR1), Model C (PPML), Model D (AB-GMM), Mixed A/C Results Log, Mixed D/C Per-Substance Routing, Mixed D/C Results Log

## Ambiguous Edges - Review These
- `Two-Model Working Spec` → `Project Status`  [AMBIGUOUS]
  PROJECT_STATUS.md · relation: references

## Knowledge Gaps
- **22 isolated node(s):** `Pipeline Run Log`, `PharmInt ETL Pipeline Reference Page`, `Hansen/J Overidentification Test`, `MASE (Mean Absolute Scaled Error)`, `Model C — Poisson Pseudo-MLE (PPML)` (+17 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 73 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Two-Model Working Spec` and `Project Status`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `_train_window()` connect `model_ab.py` to `mixed_ac.py`, `model_seasonal.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `record_comparison()` connect `mixed_ac.py` to `model_ab.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `score_counts()` connect `model_ab.py` to `mixed_ac.py`, `model_seasonal.py`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **What connects `Pipeline Run Log`, `PharmInt ETL Pipeline Reference Page`, `Hansen/J Overidentification Test` to the rest of the system?**
  _22 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `model_ab.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13131313131313133 - nodes in this community are weakly interconnected._
- **Should `mixed_ac.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11363636363636363 - nodes in this community are weakly interconnected._