# Olist Brazilian E-Commerce — Pricing Perception vs Operational Performance

## Problem Statement

What actually drives a customer's review score — the price they pay for shipping, or how reliably their order arrives?

This project answers that question by analyzing **105,363 order-item records** across roughly **96,000 delivered orders**, using the real Olist Brazilian E-Commerce dataset. Using Python and Pandas, nine raw relational tables were cleaned and joined into a single analytical table, then correlation analysis, hypothesis testing, and regression modeling were used to isolate the true driver of customer satisfaction. All analysis, reasoning, and results are documented directly inside the Jupyter notebook.

---

## Research Question

> *Does freight cost as a percentage of product price systematically predict customer review scores independent of delivery speed — and what is the relative contribution of pricing perception versus operational performance in determining customer satisfaction?*

---

## Data Source

| Attribute | Details |
|---|---|
| **Source** | Olist — Brazilian E-Commerce Public Dataset (Kaggle) |
| **Coverage** | 9 relational tables · 540+ product categories |
| **Time period** | September 2016 – October 2018 |
| **Records** | 105,363 order-item rows × 50 columns (final joined table) |
| **Population** | Real, anonymized orders placed on the Olist marketplace in Brazil |

**Dataset URL:**
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

### Important Notes

* Covers real delivered orders only — cancelled, undelivered, or incomplete orders are excluded during cleaning.
* Reviews are matched 1-to-1 with orders; duplicate review submissions on the same order are resolved by keeping the earliest response.
* Geolocation data is aggregated to zip-code level (not exact address), so distance calculations are approximate.
* 700 rows have no review score — legitimately delivered orders where the customer never left a review — and are excluded only when review score is required for an analysis step.

---

## Tools & Technologies

* **Python**
* **Pandas / NumPy**
* **Matplotlib / Seaborn**
* **SciPy** (hypothesis testing)
* **Scikit-learn** (regression)
* **Jupyter Notebook**

---

## Project Architecture

```text
Olist Raw CSV Files (9 tables)
        │
        ▼
Independent Table Profiling
(shape, dtypes, nulls, duplicates, suspicious values)
        │
        ▼
Per-Table Cleaning
(orders, reviews, geolocation, items, payments, products, category translation)
        │
        ▼
Master Analytical Table
(9 sequential joins → 105,363 rows × 50 columns)
        │
        ▼
Statistical Analysis
(correlation, hypothesis testing, regression)
```

---

## Methodology

### Phase 1 — Data Loading

All 9 raw CSV tables are loaded into memory as independent DataFrames so each can be profiled and cleaned on its own terms before anything is joined together.

### Phase 2 — Data Profiling and Cleaning

Every table is profiled first — shape, dtypes, null counts and percentages, duplicate rows, and suspicious or physically impossible values — **before** any cleaning decision is made.

#### Key Profiling Findings

| Finding | Result |
|---|---|
| Total raw orders | 99,441 |
| Total raw order items | 112,650 |
| Total raw reviews | 99,224 |
| Total raw geolocation rows | 1,000,163 |
| Unique zip codes | 19,015 |
| Unique products | 32,951 |
| Unique sellers | 3,095 |
| Duplicate reviews found | 551 |
| Rows outside Brazil (geolocation) | 31 |
| Invalid payment records | 3 |

#### Cleaning Decisions

| Issue Identified | Resolution |
|---|---|
| Non-delivered / incomplete orders | Filtered to delivered orders with complete timestamps (96,455 of 99,441 kept — 97%) |
| Duplicate reviews on the same order | Kept the earliest submission (551 duplicates removed) |
| Geolocation coordinates outside Brazil | 31 rows removed |
| ~52 coordinate rows per zip code | Aggregated to one centroid per zip (mean lat/lng, mode city/state) — 19,011 unique zips |
| Extreme delivery-time outliers | Removed using IQR-based bounds (factor = 3.0), preserving ~95.7% of cleaned orders |
| `payment_type = 'not_defined'` | 3 rows dropped |
| `payment_installments = 0` on credit card | Corrected to 1 (minimum valid value) rather than dropping the row |
| Zero-value product dimensions | Imputed using category median weight |
| Missing product category | Filled as `unknown` |
| Missing English category translations | 3 rows added manually (`pc_gamer`, a kitchen-appliances category, `unknown`) |

### Engineered Financial & Behavioral Metrics

Several derived features were engineered to power the analysis:

* `freight_ratio` (freight ÷ price) — primary variable for the research question
* `promise_gap_days`, `actual_delivery_days`, `carrier_pickup_days`, `delivery_to_customer_days`
* `delivery_status` (early / on_time / late)
* `shipping_distance_km` (haversine distance between seller and customer zip centroids)
* `is_free_shipping`, `freight_exceeds_price` (binary flags)

These metrics form the foundation for every statistical test and regression in the notebook.

---

## Master Table Design

All 9 cleaned tables are joined into a single **item-level** analytical table (one row per order-item).

```text
                 dim-like lookup tables
        products · categories · sellers · customers
                          │
                          │
   order_items ───────────┼──────── orders
   (item grain)           │       (order grain)
                          │
                          ▼
              master_analytical_table
              (105,363 rows × 50 columns)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        reviews (outcome)      payments + geolocation
```

### Join Sequence

| Join | Left | Right | Key | Type |
|---|---|---|---|---|
| 1 | orders_clean | order_items_clean | `order_id` | inner |
| 2 | master | products_clean | `product_id` | left |
| 3 | master | category_translation_clean | `product_category_name` | left |
| 4 | master | sellers | `seller_id` | left |
| 5 | master | customers | `customer_id` | left |
| 6 | master | reviews_clean | `order_id` | left |
| 7 | master | payments_agg | `order_id` | left |
| 8 | master | geo_clean (seller) | `seller_zip_code_prefix` | left |
| 9 | master | geo_clean (customer) | `customer_zip_code_prefix` | left |

### Design Decisions

* Raw source tables are never modified — all cleaning happens on copies (`*_clean` DataFrames).
* One fact row represents a unique order-item, preserving seller/product granularity while still supporting order-level aggregation for the analysis.
* The joined table is saved as a CSV checkpoint (`master_analytical_table.csv`) so later analysis phases can reload it without repeating every cleaning step and join.
* Rows with a missing review score are retained (not silently dropped) and only filtered out when a review score is actually required.

---

## Statistical Approach

* Pearson & Spearman correlation (raw and log-transformed freight ratio, plus shipping distance)
* One-way ANOVA — review score across freight ratio quartiles
* Base multivariate regression (8 features)
* Extended multivariate regression (15 features, adding category, state, weight, payment behavior as controls)

---

## Key Metrics & Results

| Metric | Result |
|---|---|
| Freight ratio vs. review score (Pearson) | r = -0.0414, p < 0.001 |
| Freight ratio vs. review score (Spearman) | r = -0.036, p < 0.001 |
| Freight ratio effect across quartiles (ANOVA) | F = 40.18, p < 0.001 |
| Base regression R² (8 features) | 0.086 |
| Extended regression R² (15 features) | 0.089 |
| Delivery timing (`is_late`) coefficient | -0.25 to -0.26 |
| Freight ratio coefficient (standardized) | +0.026 to +0.046 |

> **Note:** ANOVA's F-statistic confirms a significant difference exists between freight ratio quartiles, but F is a test statistic, not an effect size — it doesn't measure how large that difference is. No effect-size measure was calculated for this test.

---

## Notebook Structure & Insights

The analysis lives in a single Jupyter notebook, organized into 4 phases.

---

### Phase 1–3 — Data Loading, Cleaning, Master Table

Loads all 9 raw tables, profiles and cleans each independently, then joins them into the 105,363-row master analytical table described above.

#### Key Insights

* 97% of raw orders survive the "delivered with complete timestamps" filter.
* 551 duplicate reviews were found — all resolved by keeping the earliest submission.
* Geolocation required aggregating ~52 coordinate rows per zip code down to a single representative point.

> Insert Data Cleaning Summary Screenshot Here

---

### Phase 4 — Pricing Perception vs Operational Performance

#### Key Insights

* The ANOVA test is statistically significant (p < 0.001) — but the effect size was not separately calculated, so the practical size of the difference across freight ratio quartiles is unconfirmed.
* Freight ratio's correlation with review score never exceeds ~0.05 in either raw or log-transformed form.
* In both the base and extended regressions, delivery timing (`is_late`, `actual_delivery_days`) dominates the standardized coefficients — roughly **5–10x larger** in magnitude than freight ratio.
* Freight ratio's regression coefficient is small and even slightly positive after controlling for delivery performance.

> Insert Correlation & Regression Screenshot Here

---

## Key Findings

### 1. Delivery Reliability, Not Freight Pricing, Appears to Drive Satisfaction

Across correlation analysis and two regression models, delivery timing consistently shows an effect **5–10x larger** than freight ratio on customer review scores.

### 2. Freight Cost's Effect Is Statistically Significant but Small in Raw Correlation

Both correlation tests and the ANOVA reject the null hypothesis (p < 0.001), but correlation coefficients never exceed ~0.05 — significance alone doesn't confirm a large practical effect, and effect size for the ANOVA specifically was not calculated.

### 3. 700 Orders Have No Review At All

A meaningful share of delivered orders never receive a review, meaning review-based analysis inherently reflects only the subset of customers who choose to respond.

---

## Data Limitations

Understanding the limitations of the dataset is critical when interpreting the results of this analysis.

### 1. Correlational, Not Causal

All findings describe association, not causation. Late delivery is strongly *associated* with lower review scores, but unobserved factors (product quality, customer expectations, communication) could also contribute.

### 2. Review Selection Bias

Review score is only observed for customers who chose to leave a review (700 orders have none). This may not represent the opinions of silent customers.

### 3. Geolocation Is Zip-Code Level, Not Exact Address

`shipping_distance_km` is calculated between zip-code centroids, not exact delivery addresses, making it an approximation of true shipping distance.

### 4. Single-Period Analysis

This project analyzes a single ~2-year window (Sept 2016 – Oct 2018). Seasonal effects, platform growth, and policy changes over time are not modeled separately.

---

## Metric Definitions & Methodology

**FREIGHT RATIO**
*Freight value divided by product price — the primary measure of shipping cost relative to product cost for this analysis.*

**PROMISE GAP DAYS**
*Days between actual delivery date and the estimated delivery date. Negative values mean the order arrived early; positive values mean it arrived late.*

**DELIVERY STATUS**
*Categorical classification of an order as early, on_time, or late, based on promise gap days.*

**SHIPPING DISTANCE (KM)**
*Haversine (straight-line) distance between the seller's and customer's zip-code centroids.*

---

## How to Run This Project

### Prerequisites

* Python 3.9+
* Jupyter Notebook or JupyterLab
* pip

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### Step 2 — Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

### Step 3 — Open the Notebook

```bash
jupyter notebook olist_analysis_documented.ipynb
```

### Step 4 — Run All Cells

All required CSV files are already included in the repository, so the notebook can be run top to bottom with no external downloads or database setup required.

---

## Project Structure

```text
Olist_Analysis/
│
├── olist_analysis_documented.ipynb        # Main notebook — fully commented with markdown explanations
├── olist_analysis.ipynb                   # Original notebook (code only, no comments/markdown)
│
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_customers_dataset.csv
├── olist_sellers_dataset.csv
├── olist_products_dataset.csv
├── olist_geolocation_dataset.csv
├── product_category_name_translation.csv
│
├── master_analytical_table.csv            # Cleaned, joined checkpoint (generated by the notebook)
│
├── README.md
└── LICENSE
```

> **Which notebook should I open?** Start with `olist_analysis_documented.ipynb` — it contains the identical analysis and results as `olist_analysis.ipynb`, but with inline comments and markdown section headers explaining the reasoning behind every cleaning decision, join, test, and model.

---

## Skills Demonstrated

* Data Profiling
* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Statistical Hypothesis Testing (ANOVA)
* Correlation Analysis (Pearson & Spearman)
* Multivariate Regression
* Data Storytelling
* Python (Pandas, NumPy, Scikit-learn)

---

## Disclaimer

This project was created for educational and portfolio purposes.

Data originates from the publicly available Olist Brazilian E-Commerce dataset on Kaggle, released under a CC BY-NC-SA 4.0 license. All findings, interpretations, and visualizations are intended solely for analytical demonstration and should not be used for business, operational, or seller-management decisions without additional validation.
