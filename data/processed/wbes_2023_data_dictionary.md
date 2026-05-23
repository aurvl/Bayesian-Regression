# WBES 2023 Regression Dataset - Data Dictionary

Source: World Bank Enterprise Surveys public indicator dataset, 2023 survey waves only, `cut = All` and `subcut = All`.

Observation unit: one economy/survey wave. This is a simple cross-section, not a firm-level panel.

Research question proposed: Do economies where firms report higher R&D and innovation intensity also exhibit higher real annual sales growth, conditional on firm size, capital investment, exporting, access to finance, digital adoption, workforce training, certification, and infrastructure constraints?

| Processed variable | Source code | Role | Economic interpretation |
|---|---:|---|---|
| `target_real_annual_sales_growth_pct` | `perf1` | Target | Real annual sales growth (%). Continuous performance outcome used as y. |
| `alt_target_labor_productivity_growth_pct` | `perf3` | Alternative target | Real annual labor productivity growth (%). Kept for robustness / alternative notebook runs. |
| `rd_spending_firms_pct` | `t10` | R&D | Percent of firms that spent on R&D in the last fiscal year. |
| `avg_fte_workers` | `wk21` | Firm size | Average number of permanent full-time equivalent workers. |
| `log_avg_fte_workers` | `derived from wk21` | Firm size transform | Log firm-size proxy used to reduce leverage from very large average employment values. |
| `fixed_assets_purchase_firms_pct` | `perf4` | Capital investment | Percent of firms buying fixed assets. |
| `direct_exporters_10pct_sales_pct` | `tr16` | Exports | Percent of firms exporting directly at least 10% of sales. |
| `bank_loan_or_credit_line_firms_pct` | `fin14` | Access to finance | Percent of firms with a bank loan or line of credit. |
| `finance_major_constraint_firms_pct` | `fin16` | Finance constraint | Percent of firms identifying access to finance as a major or very severe constraint. |
| `product_innovation_firms_pct` | `t7` | Innovation | Percent of firms introducing a new product/service over the last 3 years. |
| `process_innovation_firms_pct` | `t9` | Innovation | Percent of firms introducing a process innovation over the last 3 years. |
| `website_firms_pct` | `t5` | Digital adoption | Percent of firms having their own website. |
| `electronic_payments_made_pct` | `bready_fin31` | Digital finance | Proportion of payments made electronically. |
| `formal_training_workers_pct` | `wk2` | Human capital | Proportion of workers offered formal training over the last fiscal year. |
| `quality_certification_firms_pct` | `bready_t1` | Quality / technology | Percent of firms with an internationally-recognized quality certification. |
| `electrical_outages_firms_pct` | `in16` | Infrastructure | Percent of firms experiencing electrical outages. |

Recommended baseline model:

`target_real_annual_sales_growth_pct ~ rd_spending_firms_pct + log_avg_fte_workers + fixed_assets_purchase_firms_pct + direct_exporters_10pct_sales_pct + bank_loan_or_credit_line_firms_pct + finance_major_constraint_firms_pct + product_innovation_firms_pct + process_innovation_firms_pct + website_firms_pct + electronic_payments_made_pct + formal_training_workers_pct`

Notes:
- `alt_target_labor_productivity_growth_pct` is kept as an alternative dependent variable.
- `avg_fte_workers` is kept for interpretability, but `log_avg_fte_workers` is recommended for regression.
- With 39 observations and many regressors, use this mainly as a teaching dataset for OLS/Bayesian regression mechanics; avoid strong causal claims.