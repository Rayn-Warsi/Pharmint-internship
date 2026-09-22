# Graph Report - Pharmint  (2026-09-02)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 200 nodes · 425 edges · 15 communities (11 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `66f75b40`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Forecast Models (A/B/D, Ensemble, GBM, LSTM)
- Mixed Model Routing (PPML/AR1/AB-GMM)
- Two-Model Spec & Results Docs
- ETL Job Pipeline (Code.py)
- Backtesting & Stability Reports
- ETL Pipeline Reference & Run Log
- Graphify Skill Documentation
- Model Comparison Aggregator
- Panel Data Builder
- Mixed Model Results Docs
- Data Pipeline Overview
- Dataset Merge Script
- Drug Substance Splitter
- Monthly Transactions Chart
- Forecast Models Tracker Page

## God Nodes (most connected - your core abstractions)
1. `_train_window()` - 22 edges
2. `main()` - 20 edges
3. `main()` - 16 edges
4. `compute_entity_effects()` - 15 edges
5. `fit_model_d_ab_gmm()` - 15 edges
6. `score_counts()` - 15 edges
7. `main()` - 15 edges
8. `load_panel()` - 14 edges
9. `split_substances()` - 14 edges
10. `oos_r2()` - 14 edges

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
- **Nine-Model Forecast Comparison Suite (A-D, E, F-GMM, G, F-LSTM)** — models_two_model_spec_model_a, models_two_model_spec_model_b, models_two_model_spec_model_c, models_two_model_spec_model_d, track_forecast_models_model_e, track_forecast_models_model_f_gmm_xgb, track_forecast_models_model_g, track_forecast_models_model_f_lstm [INFERRED 0.85]

## Communities (15 total, 4 thin omitted)

### Community 0 - "Forecast Models (A/B/D, Ensemble, GBM, LSTM)"
Cohesion: 0.12
Nodes (41): adj_r2(), ar_diagnostics(), compute_entity_effects(), fit_model_a(), fit_model_b_gmm(), fit_model_b_ols_within(), fit_model_d_ab_gmm(), fit_pooled_ols() (+33 more)

### Community 1 - "Mixed Model Routing (PPML/AR1/AB-GMM)"
Cohesion: 0.15
Nodes (24): main(), mixed_predict(), predict_a_row(), Mixed model: per-substance best-of(Model A AR(1), Model C PPML) from…, Per-row Model A predicted count: alpha_entity + y_l1*phi1, expm1'd., Pick A or C per substance by lower absolute error on the validation month…, route_by_validation(), main() (+16 more)

### Community 2 - "Two-Model Spec & Results Docs"
Cohesion: 0.13
Nodes (23): Model Ensemble Results (Drug Grain), Model Ensemble Results (Country Grain), Model GBM Results, Model LSTM Results, Model Results A-D (Drug Grain), Model Results A-D (Country Grain), Two-Model Cheatsheet, Two-Model Working Spec (+15 more)

### Community 3 - "ETL Job Pipeline (Code.py)"
Cohesion: 0.19
Nodes (21): apply_quality_filters(), clean_product_key(), enrich(), job_country_market_intel(), job_exporter_directory(), job_market_opportunities(), job_price_trends_monthly(), job_product_analytics() (+13 more)

### Community 4 - "Backtesting & Stability Reports"
Cohesion: 0.21
Nodes (14): _apply_color_scale(), canonical_name(), main(), _model_sort_key(), parse_rows(), _pivot_metric(), Backtests model_ab.py (Models A/B/C/D) and model_ensemble.py (Models F/G) at…, One row per model, one column per training window, plus stability columns. CV%… (+6 more)

### Community 5 - "ETL Pipeline Reference & Run Log"
Cohesion: 0.14
Nodes (14): country_market_intel (Code.py job output), exporter_directory (Code.py job output), market_opportunities (Code.py job output), price_trends_monthly (Code.py job output), product_analytics (Code.py job output), regional_analytics (Code.py job output), country_market_intel (Mongo aggregation collection), exporter_directory (Mongo aggregation collection) (+6 more)

### Community 6 - "Graphify Skill Documentation"
Cohesion: 0.20
Nodes (11): .claude/CLAUDE.md graphify Pointer, add-watch.md Reference, exports.md Reference, extraction-spec.md Reference, github-and-merge.md Reference, hooks.md Reference, query.md Reference, transcribe.md Reference (+3 more)

### Community 7 - "Model Comparison Aggregator"
Cohesion: 0.25
Nodes (10): _load_base(), main(), _metric_map(), Shared helper for folding a satellite model's test-set scores (Mixed (A/C),…, Rebuild model_comparision.xlsx from model_comparision.csv alone (no satellite…, metric label (as it appears in model_comparision's "metric" column) -> key in…, Load model_comparision.xlsx if it exists (stripping its NOTE row so it can be…, Update model_comparision.xlsx in place with one satellite model's scores,… (+2 more)

### Community 8 - "Panel Data Builder"
Cohesion: 0.33
Nodes (10): build_panel(), build_panel_country(), main(), _month_order(), DataFrame, Series, Build the substance x month panel used by the two-model spec (two-model-…, Distinct (year, month) pairs present in the enriched data, sorted… (+2 more)

### Community 9 - "Mixed Model Results Docs"
Cohesion: 0.57
Nodes (7): Mixed A/C Per-Substance Routing, Model A (AR1), Model C (PPML), Model D (AB-GMM), Mixed A/C Results Log, Mixed D/C Per-Substance Routing, Mixed D/C Results Log

### Community 10 - "Data Pipeline Overview"
Cohesion: 0.67
Nodes (3): Pipeline Run Log, PharmInt ETL Pipeline Reference Page, Excel → MongoDB → Dashboard Data Pipeline

## Ambiguous Edges - Review These
- `Excel → MongoDB → Dashboard Data Pipeline` → `PharmInt ETL Pipeline Reference Page`  [AMBIGUOUS]
  Track/forecast-models.html · relation: references
- `Two-Model Working Spec` → `Project Status`  [AMBIGUOUS]
  PROJECT_STATUS.md · relation: references

## Knowledge Gaps
- **21 isolated node(s):** `Pipeline Run Log`, `PharmInt ETL Pipeline Reference Page`, `PharmInt Forecast Models Reference Page`, `Hansen/J Overidentification Test`, `MASE (Mean Absolute Scaled Error)` (+16 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 69 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Excel → MongoDB → Dashboard Data Pipeline` and `PharmInt ETL Pipeline Reference Page`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Two-Model Working Spec` and `Project Status`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `_train_window()` connect `Forecast Models (A/B/D, Ensemble, GBM, LSTM)` to `Mixed Model Routing (PPML/AR1/AB-GMM)`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `record_comparison()` connect `Model Comparison Aggregator` to `Forecast Models (A/B/D, Ensemble, GBM, LSTM)`, `Mixed Model Routing (PPML/AR1/AB-GMM)`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Why does `predict_ppml()` connect `Mixed Model Routing (PPML/AR1/AB-GMM)` to `Forecast Models (A/B/D, Ensemble, GBM, LSTM)`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **What connects `Pipeline Run Log`, `PharmInt ETL Pipeline Reference Page`, `PharmInt Forecast Models Reference Page` to the rest of the system?**
  _21 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Forecast Models (A/B/D, Ensemble, GBM, LSTM)` be split into smaller, more focused modules?**
  _Cohesion score 0.12303422756706753 - nodes in this community are weakly interconnected._