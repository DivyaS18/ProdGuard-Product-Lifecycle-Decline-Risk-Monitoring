FMCG Product Lifecycle, Decline Risk Monitoring
Project Overview

This repository contains an end-to-end, large-scale FMCG analytics and decision intelligence system designed to analyze product performance, lifecycle stages, decline risk, pricing impact, customer buying behavior, and future revenue risk using rule-based analytics and machine learning.

The project simulates real-world FMCG and retail analytics workflows, operating on multi-million-row datasets and producing executive-ready insights for product, category, and leadership teams.

Business Objectives

Identify product lifecycle stages (Introduction, Growth, Maturity, Decline)

Detect early warning signals of SKU-level decline

Quantify decline risk and revenue at risk

Understand pricing and promotion dependency

Forecast risk-adjusted future sales

Discover customer co-purchase patterns for declining products

Recommend data-driven bundles and recovery actions

Datasets Used:
1) FMCG Sales Dataset

Size: ~1.1 million rows × 33 columns

This dataset captures daily SKU-level sales, pricing, inventory, supply chain, geographic, and weather context across multiple countries and channels.

Key Columns Include:

Time & Calendar: date, year, month, weekofyear, weekday, is_weekend, is_holiday

Product Hierarchy: sku_id, sku_name, category, subcategory, brand

Sales & Pricing: units_sold, list_price, discount_pct, promo_flag

Revenue: gross_sales, net_sales, margin_pct

Inventory & Supply Chain: stock_on_hand, stock_out_flag, lead_time_days

Geography & Weather: country, city, latitude, longitude, temperature, rain_mm

2️) Transaction Dataset

Size: ~12.1 million rows × 20 columns

This dataset captures transaction-level customer purchase behavior, enabling advanced market basket analysis and bundle recommendation systems.

Key Columns Include:

Transaction & Customer: transaction_id, customer_id

Product Hierarchy: sku_id, sku_name, category, subcategory, brand

Purchase Details: quantity, list_price, discount_pct, promo_flag

Context: date, store_id, country, city, channel

Dataset Availability Notice

Due to GitHub file size limitations, the full datasets (~1.88 GB) are not included in this repository.

What is provided

Sample datasets for reproducibility

All data processing, modeling, and analysis code

Exported analytical outputs

Full datasets

Can be shared upon request for academic, evaluation, or interview purposes

Project Architecture & Analytical Flow
Step 0 – Data Preprocessing

Data cleaning & validation

Time ordering and aggregation

Feature consistency checks

Step 1 – Sales Performance Analysis

Overall, country-wise, brand-wise, and SKU-wise performance

Revenue contribution & Pareto (80/20) analysis

Step 2 – Trends & Seasonality

Monthly aggregation

Seasonal pattern detection

Growth rate and volatility analysis

Step 3 – Product Lifecycle Modeling

Product age calculation

Rolling averages and trend smoothing

Peak sales and drop-from-peak detection

Lifecycle stage classification:

Introduction

Growth

Maturity

Decline

Rule-based decline risk scoring

Step 4 – Machine Learning Early Warning System

Time-aware target construction (no data leakage)

Logistic Regression (explainable baseline)

Random Forest (non-linear signal detection)

ML-based decline probability and hybrid risk flags

Step 5 – Executive Monitoring Framework

SKU health KPIs

Portfolio health summary

Priority SKU identification

Lifecycle- and risk-based action recommendations

Step 5B – Pricing & Promotion Impact Analysis

Discount dependency detection

Promotion intensity measurement

Margin pressure indicators

Post-promotion demand drop analysis

Decline root-cause classification

Step 6 – Forecast-Driven Business Outlook

Risk-adjusted sales forecasts

Best / Base / Worst-case scenarios

Revenue-at-risk estimation

Step 7 – Market Basket Analysis

FP-Growth on 12M+ transactions

Decline-anchored association rules

Cross-sell and recovery opportunities

Step 8 – Combo & Bundle Recommendation System

Declining SKUs as anchor products

Bundle scoring using support, confidence, and lift

Executive-ready bundle recommendations

Key Outputs

SKU lifecycle classification

Decline risk scores & early warning flags

Priority SKU lists for management focus

Pricing risk & root-cause insights

Revenue-at-risk forecasts

Market basket association rules

Bundle & combo recommendations

Tools & Technologies

Python: Pandas, NumPy, Matplotlib

Machine Learning: Scikit-learn

Association Rules: MLxtend (FP-Growth)

Visualization & BI: Power BI

Large-scale data handling: 1M+ and 12M+ row datasets

Who This Project Is For

Data Analyst / Data Scientist roles

FMCG & Retail Analytics teams

Product & Category Managers

Business Intelligence professionals

Advanced analytics academic evaluation (CDAC / PG-level)


This repository demonstrates

Strong data analytics foundations

Business-driven machine learning

Scalable, real-world project design

Executive-level insight generation
