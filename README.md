# Olist Brazilian E-Commerce — Customer Satisfaction & Seller Lifecycle Intelligence

## Problem Statement

What actually drives a customer's review score — the price they pay for shipping, or how reliably their order arrives? And can a marketplace tell that a seller is starting to decline *before* customers start punishing them in reviews?

This project answers these questions by analyzing **105,363 order-item records** across **~92,000 orders**, **2,915 sellers**, and **32,951 products**, using the real Olist Brazilian E-Commerce dataset. Using Python and Pandas, nine raw relational tables were cleaned and joined into a single analytical table, then statistical hypothesis testing and machine learning were used to isolate the true drivers of customer satisfaction and the earliest behavioral signal of seller decline. All analysis, reasoning, and results are documented directly inside the Jupyter notebook.

---

## Research Questions

### Question 1 — Pricing Perception vs Operational Performance
> *Does freight cost as a percentage of product price systematically predict customer review scores independent of delivery speed — and what is the relative contribution of pricing perception versus operational performance in determining customer satisfaction?*

### Question 2 — Seller Lifecycle Trust Erosion
> *Do Olist sellers follow a predictable lifecycle of trust erosion, where early operational decisions create compounding effects on their long-term review trajectory — and can we identify a leading behavioral signal of decline before customers begin punishing them in reviews?*

---

## Data Source

| Attribute | Details |
|---|---|
| **Source** | Olist — Brazilian E-Commerce Public Dataset (Kaggle) |
| **Coverage** | 2,906+ sellers · 9 relational tables · 540+ product categories |
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
* **Scikit-learn** (regression & classification)
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
Q1: Statistical Analysis
(correlation, hypothesis testing, regression)
        │
        ▼
Q2: Seller Lifecycle Analysis
(early/mid/late phase segmentation, archetype classification)
        │
        ▼
Predictive Modeling
(Logistic Regression + Random Forest, cross-validated)
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

* `freight_ratio` (freight ÷ price) — primary variable for Q1
* `promise_gap_days`, `actual_delivery_days`, `carrier_pickup_days`, `delivery_to_customer_days`
* `delivery_status` (early / on_time / late)
* `shipping_distance_km` (haversine distance between seller and customer zip centroids)
* `is_free_shipping`, `freight_exceeds_price` (binary flags)
* `lifecycle_phase` (early / mid / late, per seller) and `seller_archetype` (deteriorating / stable / improving)

These metrics form the foundation for every statistical test, regression, and model in the notebook.

---

## Master Table Design

All 9 cleaned tables are joined into a single **item-level** analytical table (one row per order-item), preserving the granularity needed for both research questions.

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
* One fact row represents a unique order-item, preserving seller/product granularity for Q2 while still supporting order-level aggregation for Q1.
* The joined table is saved as a CSV checkpoint (`master_analytical_table.csv`) so later analysis phases can reload it without repeating every cleaning step and join.
* Rows with a missing review score are retained (not silently dropped) and only filtered out when a review score is actually required.

---

## Statistical & Modeling Approach

### Q1 — Pricing Perception vs Operational Performance

* Pearson & Spearman correlation (raw and log-transformed freight ratio, plus shipping distance)
* One-way ANOVA — review score across freight ratio quartiles
* Kruskal-Wallis — non-parametric equivalent of the ANOVA
* Mann-Whitney U — early vs. late delivery review scores
* Base multivariate regression (8 features)
* Extended multivariate regression (15 features, adding category, state, weight, payment behavior as controls)

### Q2 — Seller Lifecycle Trust Erosion

* Filtered to sellers with **≥15 orders** (941 qualifying sellers, covering 90% of all reviewed orders — threshold chosen by testing seller survival at 5/10/15/20/30/50-order cutoffs)
* Each seller's order history split into **early / mid / late thirds**
* Sellers classified into 3 archetypes based on review score change (late minus early):
  * `deteriorating` — change ≤ -0.5
  * `improving` — change ≥ +0.5
  * `stable` — everything in between
* Early/mid/late phase behavior compared across archetypes to isolate a leading signal
* Seller tenure compared across archetypes as an additional control

### Predictive Modeling

Using only early/mid-phase behavioral features (no late-phase data, since the goal is early prediction):

* **Logistic Regression** (scaled features, `class_weight='balanced'`)
* **Random Forest** (raw features, `class_weight='balanced'`)
* Evaluated on a held-out stratified test set and validated with **5-fold stratified cross-validation**

---

## Key Metrics & Results

### Core Statistical Results

| Metric | Result |
|---|---|
| Freight ratio vs. review score (correlation) | r ≈ -0.036 to -0.041 |
| Freight ratio effect across quartiles (ANOVA) | F = 40.18, p < 0.001 |
| Early vs. late delivery (Mann-Whitney U) | U = 325.1M, p < 0.001 |
| Base regression R² (8 features) | 0.086 |
| Extended regression R² (15 features) | 0.089 |
| Delivery timing (`is_late`) coefficient | -0.25 to -0.26 |
| Freight ratio coefficient (standardized) | +0.03 to +0.05 |

### Seller Lifecycle Results

| Metric | Result |
|---|---|
| Qualifying sellers (≥15 orders) | 941 (90% of reviewed orders) |
| Deteriorating sellers | 144 (15.3%) |
| Stable sellers | 658 (69.9%) |
| Improving sellers | 139 (14.8%) |
| Deteriorating: early → late review score | 4.500 → 3.677 (-0.823) |
| Improving: early → late review score | 3.709 → 4.545 (+0.836) |
| Deteriorating mid-phase late-delivery jump | +2.22 pts (vs. +1.84 stable, -1.94 improving) |

### Predictive Model Results

| Model | ROC-AUC (test) | ROC-AUC (5-fold CV mean ± std) |
|---|---|---|
| Logistic Regression | 0.709 | **0.716 ± 0.017** |
| Random Forest | 0.745 | 0.681 ± 0.036 |

---

## Notebook Structure & Insights

The analysis lives in a single Jupyter notebook, organized into 6 phases.

---

### Phase 1–3 — Data Loading, Cleaning, Master Table

Loads all 9 raw tables, profiles and cleans each independently, then joins them into the 105,363-row master analytical table described above.

#### Key Insights

* 97% of raw orders survive the "delivered with complete timestamps" filter.
* 551 duplicate reviews were found — all resolved by keeping the earliest submission.
* Geolocation required aggregating ~52 coordinate rows per zip code down to a single representative point.

> Insert Data Cleaning Summary Screenshot Here

---

### Phase 4 — Q1: Pricing Perception vs Operational Performance

#### Key Insights

* All three hypothesis tests (ANOVA, Kruskal-Wallis, Mann-Whitney U) are statistically significant (p < 0.001) — but the effect sizes are consistently tiny.
* Freight ratio's correlation with review score never exceeds ~0.05 in either raw or log-transformed form.
* In both the base and extended regressions, delivery timing (`is_late`, `actual_delivery_days`) dominates the standardized coefficients — roughly **5–10x larger** in magnitude than freight ratio.
* Freight ratio's regression coefficient is small and even slightly positive after controlling for delivery performance.

> Insert Q1 Correlation & Regression Screenshot Here

---

### Phase 5 — Q2: Seller Lifecycle Trust Erosion

#### Key Insights

* Deteriorating sellers start with the *highest* early-phase review score (4.500) of any archetype — they look no different, or even better, than stable sellers at first.
* The earliest detectable warning sign is an accelerating late-delivery rate in the **mid** phase — before the review score itself has visibly declined.
* Freight ratio change is small and similar across all three archetypes — pricing behavior is **not** a leading signal of seller decline.

> Insert Seller Lifecycle Trajectory Screenshot Here

---

### Phase 6 — Predictive Modeling

#### Key Insights

* Logistic Regression is both stronger and more stable under cross-validation than Random Forest, despite scoring slightly lower on a single test split.
* The mid-phase late-delivery-rate change and early-phase review score are the most influential features in both models.

> Insert Feature Importance Screenshot Here

---

## Key Findings

### 1. Delivery Reliability, Not Freight Pricing, Drives Satisfaction

Across correlation analysis, hypothesis testing, and two regression models, delivery timing consistently shows an effect **5–10x larger** than freight ratio on customer review scores.

### 2. Freight Cost Has a Statistically Significant but Practically Negligible Effect

All three hypothesis tests reject the null hypothesis (p < 0.001), but correlation coefficients never exceed ~0.05 — a textbook example of statistical significance without practical significance at this scale of data.

### 3. Deteriorating Sellers Start Out Looking *Better* Than Average

Sellers who eventually deteriorate have the highest early-phase review scores (4.500) of any archetype — making review score alone a poor early-warning indicator.

### 4. Late-Delivery Rate Acceleration Is the Leading Signal of Seller Decline

Deteriorating sellers show the sharpest early-to-mid increase in late-delivery percentage (+2.22 points), well ahead of any visible drop in review score.

### 5. Pricing Behavior Is Not a Leading Signal

Freight ratio changes are small and similar across deteriorating, stable, and improving sellers — ruling out pricing as an early indicator of decline.

### 6. A Simple Logistic Regression Outperforms Random Forest on Stability

Despite a lower single-split score, Logistic Regression generalizes more reliably across cross-validation folds (0.716 ± 0.017 vs. 0.681 ± 0.036), making it the more trustworthy model for a real early-warning system.

### 7. 90% of Reviewed Orders Come From Just 941 Sellers

Filtering to sellers with 15+ orders retains 90% of order volume while keeping enough history per seller to detect a lifecycle trend — a practical sample-size trade-off.

### 8. 700 Orders Have No Review At All

A meaningful share of delivered orders never receive a review, meaning review-based analyses inherently reflect only the subset of customers who choose to respond.

---

## Data Limitations

Understanding the limitations of the dataset is critical when interpreting the results of this analysis.

### 1. Correlational, Not Causal

All findings describe association, not causation. Late delivery is strongly *associated* with lower review scores, but unobserved factors (product quality, customer expectations, communication) could also contribute.

### 2. Review Selection Bias

Review score is only observed for customers who chose to leave a review (700 orders have none). This may not represent the opinions of silent customers.

### 3. Geolocation Is Zip-Code Level, Not Exact Address

`shipping_distance_km` is calculated between zip-code centroids, not exact delivery addresses, making it an approximation of true shipping distance.

### 4. Minimum-Order Threshold Trade-off

The 15-order cutoff used for Q2 balances sample size against having enough history per seller, but excludes smaller/newer sellers who may show different lifecycle patterns.

### 5. Modest Predictive Performance

The deterioration classifier's ROC-AUC (~0.72) reflects a real but moderate signal — useful for prioritization and monitoring, not for high-stakes automated decisions about individual sellers.

### 6. Single-Period Analysis

This project analyzes a single ~2-year window (Sept 2016 – Oct 2018). Seasonal effects, platform growth, and policy changes over time are not modeled separately.

---

## Metric Definitions & Methodology

**FREIGHT RATIO**
*Freight value divided by product price — the primary measure of shipping cost relative to product cost for Q1.*

**PROMISE GAP DAYS**
*Days between actual delivery date and the estimated delivery date. Negative values mean the order arrived early; positive values mean it arrived late.*

**DELIVERY STATUS**
*Categorical classification of an order as early, on_time, or late, based on promise gap days.*

**SHIPPING DISTANCE (KM)**
*Haversine (straight-line) distance between the seller's and customer's zip-code centroids.*

**LIFECYCLE PHASE**
*Early, mid, or late third of a seller's chronological order history, used to detect trends over a seller's tenure on the platform.*

**SELLER ARCHETYPE**
*Classification of a seller as deteriorating, stable, or improving, based on the change in average review score from their early phase to their late phase.*

**REVIEW SCORE CHANGE**
*A seller's average late-phase review score minus their average early-phase review score.*

**LATE DELIVERY %**
*Percentage of a seller's orders (within a given lifecycle phase) that arrived after the estimated delivery date.*

**ROC-AUC**
*Area under the receiver operating characteristic curve — a measure of how well a classification model distinguishes deteriorating sellers from non-deteriorating ones, independent of any single probability threshold.*

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
jupyter notebook olist_analysis_documented_full.ipynb
```

### Step 4 — Run All Cells

All required CSV files are already included in the repository, so the notebook can be run top to bottom with no external downloads or database setup required.

---

## Project Structure

```text
Olist_Analysis/
│
├── olist_analysis_documented_full.ipynb   # Main notebook — fully commented with markdown explanations
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

> **Which notebook should I open?** Start with `olist_analysis_documented_full.ipynb` — it contains the identical analysis and results as `olist_analysis.ipynb`, but with inline comments and markdown section headers explaining the reasoning behind every cleaning decision, join, test, and model.

---

## Skills Demonstrated

* Data Profiling
* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Statistical Hypothesis Testing (ANOVA, Kruskal-Wallis, Mann-Whitney U)
* Correlation Analysis (Pearson & Spearman)
* Multivariate Regression
* Customer Behavior Segmentation
* Predictive Modeling (Logistic Regression, Random Forest)
* Model Validation (Cross-Validation, ROC-AUC)
* Data Storytelling
* Python (Pandas, NumPy, Scikit-learn)

---

## Disclaimer

This project was created for educational and portfolio purposes.

Data originates from the publicly available Olist Brazilian E-Commerce dataset on Kaggle, released under a CC BY-NC-SA 4.0 license. All findings, interpretations, and visualizations are intended solely for analytical demonstration and should not be used for business, operational, or seller-management decisions without additional validation.
