import pandas as pd

fmcg=pd.read_csv('fmcg_sales_3years_1M_rows.csv')

# STEP 1.1 – Overall Sales Snapshot

print(fmcg.shape)

total_units = fmcg['units_sold'].sum()
total_gross_sales = fmcg['gross_sales'].sum()
total_net_sales = fmcg['net_sales'].sum()
avg_list_price = fmcg['list_price'].mean()
avg_discount = fmcg['discount_pct'].mean()

print("\nOVERALL SALES PERFORMANCE")
print("-------------------------")
print(f"Total Units Sold        : {total_units:,.0f}")
print(f"Total Gross Sales       : ₹{total_gross_sales:,.2f}")
print(f"Total Net Sales         : ₹{total_net_sales:,.2f}")
print(f"Average List Price      : ₹{avg_list_price:.2f}")
print(f"Average Discount (%)    : {avg_discount:.2f}")


# STEP 1.2 – Country-wise Sales Performance

country_sales = (
    fmcg
    .groupby('country')
    .agg(
        total_units=('units_sold', 'sum'),
        total_net_sales=('net_sales', 'sum')
    )
    .sort_values('total_net_sales', ascending=False)
)

country_sales['revenue_contribution_%'] = (
    country_sales['total_net_sales'] /
    country_sales['total_net_sales'].sum() * 100
)

print(country_sales)


#Changing Brand Names
# Step 1.x – Create display-friendly brand names

brand_mapping = {
    'BrandA': 'FreshNest',
    'BrandB': 'PureDrop',
    'BrandC': 'DailyJoy',
    'BrandD': 'HomePlus',
    'BrandE': 'NutriLife',
    'BrandF': 'AquaLeaf'
}

# New column for visualization & reporting only
fmcg['brand_display'] = fmcg['brand'].map(brand_mapping)

# Quick validation
print(fmcg[['brand', 'brand_display']].drop_duplicates())


# Create display-friendly SKU names using new brand names

fmcg['sku_display_name'] = (
    fmcg['sku_name']
    .replace(brand_mapping, regex=True)
)

# Validate
print(fmcg[['sku_name', 'sku_display_name']].drop_duplicates().head(10))



# STEP 1.3 – Brand-wise Sales Performance using display names

brand_sales = (
    fmcg
    .groupby('brand_display')
    .agg(
        total_units=('units_sold', 'sum'),
        total_net_sales=('net_sales', 'sum')
    )
    .sort_values('total_net_sales', ascending=False)
)

brand_sales['revenue_contribution_%'] = (
    brand_sales['total_net_sales'] /
    brand_sales['total_net_sales'].sum() * 100
)

print(brand_sales.head(15))



# STEP 1.4 – SKU-wise Sales Performance (Top Products)

sku_sales = (
    fmcg
    .groupby(['sku_id', 'sku_display_name'])
    .agg(
        total_units=('units_sold', 'sum'),
        total_net_sales=('net_sales', 'sum')
    )
    .sort_values('total_net_sales', ascending=False)
)

print(sku_sales.head(10))

# STEP 1.5 – SKU-level Revenue Concentration (Pareto Analysis)

sku_sales_sorted = sku_sales.copy()

sku_sales_sorted['cumulative_revenue_%'] = (
    sku_sales_sorted['total_net_sales'].cumsum() /
    sku_sales_sorted['total_net_sales'].sum() * 100
)

# Revenue contribution of top 20% SKUs
top_20_skus = int(0.2 * len(sku_sales_sorted))
top_20_revenue_share = (
    sku_sales_sorted
    .head(top_20_skus)['total_net_sales']
    .sum() /
    sku_sales_sorted['total_net_sales'].sum() * 100
)

print(f"Top 20% SKUs contribute {top_20_revenue_share:.2f}% of total net sales")



#Visual Representation
import matplotlib.pyplot as plt

# Net sales by country
country_sales_sorted = country_sales.sort_values(
    'total_net_sales', ascending=True
)

plt.figure()
plt.barh(
    country_sales_sorted.index,
    country_sales_sorted['total_net_sales']
)
plt.xlabel("Total Net Sales")
plt.ylabel("Country")
plt.title("Net Sales by Country")
plt.tight_layout()
plt.show()


# Net sales by brand
brand_sales_sorted = brand_sales.sort_values(
    'total_net_sales', ascending=True
)

plt.figure()
plt.barh(
    brand_sales_sorted.index,
    brand_sales_sorted['total_net_sales']
)
plt.xlabel("Total Net Sales")
plt.ylabel("Brand")
plt.title("Net Sales by Brand")
plt.tight_layout()
plt.show()




# Top 10 SKUs by net sales
top_10_skus = sku_sales.head(10)

plt.figure()
plt.barh(
    top_10_skus.index.get_level_values('sku_display_name'),
    top_10_skus['total_net_sales']
)
plt.xlabel("Total Net Sales")
plt.ylabel("SKU")
plt.title("Top 10 SKUs by Net Sales")
plt.tight_layout()
plt.show()


# Cumulative revenue contribution curve
plt.figure()
plt.plot(
    sku_sales_sorted['cumulative_revenue_%'].values
)
plt.xlabel("Number of SKUs (sorted by revenue)")
plt.ylabel("Cumulative Revenue %")
plt.title("SKU Revenue Concentration Curve")
plt.tight_layout()
plt.show()



# STEP 2.1 – Monthly Sales Aggregation 

monthly_sales = (
    fmcg
    .groupby(['year', 'month'])
    .agg(
        total_units=('units_sold', 'sum'),
        total_net_sales=('net_sales', 'sum'),
        avg_discount=('discount_pct', 'mean')
    )
    .reset_index()
    .sort_values(['year', 'month'])
)

print(monthly_sales.head())
print(monthly_sales.tail())


#STEP 2.2 - Visualization

# Create a plotting-friendly datetime index (non-destructive)
monthly_sales['year_month'] = pd.to_datetime(
    monthly_sales['year'].astype(str) + '-' +
    monthly_sales['month'].astype(str) + '-01'
)


# STEP 2.2.2 — Net Sales Trend Over Time
plt.figure()
plt.plot(
    monthly_sales['year_month'],
    monthly_sales['total_net_sales']
)
plt.xlabel("Time (Year-Month)")
plt.ylabel("Total Net Sales")
plt.title("Monthly Net Sales Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#STEP 2.2.3 — Units Sold Trend Over Time (Volume View)
plt.figure()
plt.plot(
    monthly_sales['year_month'],
    monthly_sales['total_units']
)
plt.xlabel("Time (Year-Month)")
plt.ylabel("Total Units Sold")
plt.title("Monthly Units Sold Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# STEP 2.3.1 – Month-wise Average Seasonality

monthly_seasonality = (
    monthly_sales
    .groupby('month')
    .agg(
        avg_monthly_units=('total_units', 'mean'),
        avg_monthly_net_sales=('total_net_sales', 'mean')
    )
    .reset_index()
    .sort_values('month')
)

print(monthly_seasonality)


#STEP 2.3.2 — Visualize Seasonal Pattern (Net Sales)

plt.figure()
plt.plot(
    monthly_seasonality['month'],
    monthly_seasonality['avg_monthly_net_sales'],
    marker='o'
)
plt.xlabel("Month")
plt.ylabel("Average Monthly Net Sales")
plt.title("Seasonality Pattern – Average Net Sales by Month")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


#STEP 2.3.3 — Visualize Seasonal Pattern (Units Sold)
plt.figure()
plt.plot(
    monthly_seasonality['month'],
    monthly_seasonality['avg_monthly_units'],
    marker='o'
)
plt.xlabel("Month")
plt.ylabel("Average Monthly Units Sold")
plt.title("Seasonality Pattern – Average Units Sold by Month")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


#STEP 2.3.4 Identify peak and low season months

peak_month = monthly_seasonality.loc[
    monthly_seasonality['avg_monthly_net_sales'].idxmax()
]

low_month = monthly_seasonality.loc[
    monthly_seasonality['avg_monthly_net_sales'].idxmin()
]

print("PEAK SEASON MONTH")
print(peak_month)

print("\nLOW SEASON MONTH")
print(low_month)


# STEP 2.4.1 – Month-over-Month Growth Rate Calculation

# Growth rate for Net Sales (MoM % change)
monthly_sales['net_sales_mom_growth_pct'] = (
    monthly_sales['total_net_sales']
    .pct_change() * 100
)

# Growth rate for Units Sold (MoM % change)
monthly_sales['units_mom_growth_pct'] = (
    monthly_sales['total_units']
    .pct_change() * 100
)

# Inspect first few rows (first month will naturally be NaN)
print(monthly_sales[['year', 'month',
                      'net_sales_mom_growth_pct',
                      'units_mom_growth_pct']].head(6))



# STEP 2.4.2 – Net Sales Month-over-Month Growth Trend

plt.figure()

# Plot MoM growth rate for net sales
plt.plot(
    monthly_sales['year_month'],
    monthly_sales['net_sales_mom_growth_pct']
)

# Reference line at 0% growth (important for interpretation)
plt.axhline(0)

plt.xlabel("Time (Year-Month)")
plt.ylabel("Net Sales Growth (%)")
plt.title("Month-over-Month Net Sales Growth Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 2.4.3 – Units Sold Month-over-Month Growth Trend

plt.figure()

# Plot MoM growth rate for units sold
plt.plot(
    monthly_sales['year_month'],
    monthly_sales['units_mom_growth_pct']
)

# Reference line at 0% growth
plt.axhline(0)

plt.xlabel("Time (Year-Month)")
plt.ylabel("Units Sold Growth (%)")
plt.title("Month-over-Month Units Sold Growth Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 2.4.4 – Growth Volatility Metrics

avg_net_sales_growth = monthly_sales['net_sales_mom_growth_pct'].mean()
std_net_sales_growth = monthly_sales['net_sales_mom_growth_pct'].std()

avg_units_growth = monthly_sales['units_mom_growth_pct'].mean()
std_units_growth = monthly_sales['units_mom_growth_pct'].std()

print("GROWTH RATE SUMMARY")
print("-------------------")
print(f"Average MoM Net Sales Growth (%) : {avg_net_sales_growth:.2f}")
print(f"Net Sales Growth Volatility (Std): {std_net_sales_growth:.2f}")

print(f"\nAverage MoM Units Growth (%)     : {avg_units_growth:.2f}")
print(f"Units Growth Volatility (Std)    : {std_units_growth:.2f}")


# STEP 3.1 – SKU-level Monthly Aggregation

sku_monthly = (
    fmcg
    .groupby(['sku_id', 'sku_display_name', 'year', 'month'])
    .agg(
        total_units=('units_sold', 'sum'),
        total_net_sales=('net_sales', 'sum'),
        avg_discount=('discount_pct', 'mean'),
        promo_frequency=('promo_flag', 'mean')  # share of promo days
    )
    .reset_index()
    .sort_values(['sku_id', 'year', 'month'])
)

print(sku_monthly.head())
print(sku_monthly.tail())


# STEP 3.2 – Product Age (Months Since Launch)

# Create a datetime index for safe ordering
sku_monthly['year_month'] = pd.to_datetime(
    sku_monthly['year'].astype(str) + '-' +
    sku_monthly['month'].astype(str) + '-01'
)

# First appearance per SKU
sku_monthly['launch_date'] = (
    sku_monthly
    .groupby('sku_id')['year_month']
    .transform('min')
)

# Product age in months
sku_monthly['product_age_months'] = (
    (sku_monthly['year_month'].dt.year - sku_monthly['launch_date'].dt.year) * 12 +
    (sku_monthly['year_month'].dt.month - sku_monthly['launch_date'].dt.month)
)

print(sku_monthly[['sku_display_name',
                   'year_month',
                   'product_age_months']].head(10))


# STEP 3.3 – SKU-level MoM Growth

sku_monthly['sales_growth_pct'] = (
    sku_monthly
    .groupby('sku_id')['total_net_sales']
    .pct_change() * 100
)


# STEP 3.4 – Rolling Average Sales (3-month window)

sku_monthly['rolling_avg_sales_3m'] = (
    sku_monthly
    .groupby('sku_id')['total_net_sales']
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)


# STEP 3.5 – Peak Sales Detection

sku_monthly['peak_sales'] = (
    sku_monthly
    .groupby('sku_id')['total_net_sales']
    .transform('max')
)

# Drop from peak (%)
sku_monthly['drop_from_peak_pct'] = (
    (sku_monthly['total_net_sales'] - sku_monthly['peak_sales']) /
    sku_monthly['peak_sales'] * 100
)


# STEP 3.6 – Consecutive Negative Growth Counter

def consecutive_negative_growth(series):
    count = 0
    counts = []
    for value in series:
        if value < 0:
            count += 1
        else:
            count = 0
        counts.append(count)
    return counts

sku_monthly['consecutive_negative_growth'] = (
    sku_monthly
    .groupby('sku_id')['sales_growth_pct']
    .transform(consecutive_negative_growth)
)


# STEP 3.7 – Sales Volatility (Std Dev)

sku_monthly['sales_std_dev'] = (
    sku_monthly
    .groupby('sku_id')['total_net_sales']
    .transform('std')
)

sku_monthly['coefficient_of_variation'] = (
    sku_monthly['sales_std_dev'] /
    sku_monthly
    .groupby('sku_id')['total_net_sales']
    .transform('mean')
)

# Do not evaluate drop-from-peak during Introduction stage
sku_monthly.loc[
    sku_monthly['product_age_months'] <= 6,
    'drop_from_peak_pct'
] = 0


# STEP 3.8 – Lifecycle Stage Assignment (Refined, 4-Stage PLC)

# Calculate rolling trend direction
sku_monthly['rolling_trend'] = (
    sku_monthly
    .groupby('sku_id')['rolling_avg_sales_3m']
    .diff()
)

def lifecycle_stage(row):

    # 1. Introduction: early life
    if row['product_age_months'] <= 6:
        return 'Introduction'

    # 2. Decline: sustained deterioration
    elif (row['drop_from_peak_pct'] <= -10) and (row['consecutive_negative_growth'] >= 3):
        return 'Decline'

    # 3. Growth: sustained upward trend (smoothed)
    elif row['rolling_trend'] > 0:
        return 'Growth'

    # 4. Maturity: plateau or slow movement near peak
    else:
        return 'Maturity'


sku_monthly['lifecycle_stage'] = sku_monthly.apply(
    lifecycle_stage, axis=1
)

sku_monthly['rolling_trend'] = sku_monthly['rolling_trend'].fillna(0)

sku_monthly.to_csv('sku_monthly.csv',index=False)


# Check for duplicate column names
dup_cols = sku_monthly.columns[sku_monthly.columns.duplicated()]
print("Duplicate columns:", dup_cols.tolist())

# Remove duplicate columns, keep the first occurrence
sku_monthly = sku_monthly.loc[:, ~sku_monthly.columns.duplicated()]

print(sku_monthly.columns.is_unique)


#STEP 3.9.1 – Normalize 
from sklearn.preprocessing import MinMaxScaler

risk_cols = [
    'drop_from_peak_pct',
    'consecutive_negative_growth',
    'coefficient_of_variation'
]

risk_features = sku_monthly[risk_cols].fillna(0)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(risk_features)

sku_monthly['drop_from_peak_scaled'] = scaled[:, 0]
sku_monthly['neg_growth_scaled'] = scaled[:, 1]
sku_monthly['volatility_scaled'] = scaled[:, 2]


#STEP 3.9.2 - Decline Risk Score (Introduction excluded)
sku_monthly['decline_risk_score'] = 0.0

mask = sku_monthly['lifecycle_stage'] != 'Introduction'

sku_monthly.loc[mask, 'decline_risk_score'] = (
    0.4 * sku_monthly.loc[mask, 'drop_from_peak_scaled'] +
    0.4 * sku_monthly.loc[mask, 'neg_growth_scaled'] +
    0.2 * sku_monthly.loc[mask, 'volatility_scaled']
)


#STEP 3.10 – Risk Bucketing
def risk_bucket(row):
    if row['lifecycle_stage'] == 'Introduction':
        return 'Low Risk'
    elif row['decline_risk_score'] >= 0.7:
        return 'High Risk'
    elif row['decline_risk_score'] >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

sku_monthly['decline_risk_level'] = sku_monthly.apply(risk_bucket, axis=1)

#checking
# 1. Column uniqueness
print("Columns unique:", sku_monthly.columns.is_unique)

# 2. Lifecycle vs risk sanity
print(
    sku_monthly.groupby('lifecycle_stage')['decline_risk_level']
    .value_counts()
)

# 3. Latest snapshot
latest_sku_status = (
    sku_monthly
    .sort_values(['sku_id', 'year_month'])
    .groupby('sku_id')
    .tail(1)
)

print(latest_sku_status[['sku_display_name',
                         'lifecycle_stage',
                         'decline_risk_score',
                         'decline_risk_level']]
      .sort_values('decline_risk_score', ascending=False))


# STEP 3.11.1 – Sales Lifecycle with Rolling Average (Context)

sku_name = 'FreshNest Soda'
sku_data = sku_monthly[sku_monthly['sku_display_name'] == sku_name]

plt.figure()

# Raw monthly sales
plt.plot(
    sku_data['year_month'],
    sku_data['total_net_sales'],
    label='Monthly Net Sales'
)

# Smoothed sales
plt.plot(
    sku_data['year_month'],
    sku_data['rolling_avg_sales_3m'],
    label='3-Month Rolling Average'
)

plt.xlabel("Time")
plt.ylabel("Net Sales")
plt.title(f"Sales Lifecycle – {sku_name}")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# STEP 3.11.2 – Lifecycle Stage Assignment (Updated Logic)

plt.figure()

for stage in ['Introduction', 'Growth', 'Maturity', 'Decline']:
    stage_data = sku_data[sku_data['lifecycle_stage'] == stage]
    plt.scatter(
        stage_data['year_month'],
        stage_data['total_net_sales'],
        label=stage
    )

plt.xlabel("Time")
plt.ylabel("Net Sales")
plt.title(f"Lifecycle Stage Assignment – {sku_name}")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()



# STEP 4.1 – Latest SKU Snapshot (one row per SKU)

latest_sku_status = (
    sku_monthly
    .sort_values(['sku_id', 'year_month'])
    .groupby('sku_id')
    .tail(1)
    .reset_index(drop=True)
)

print(latest_sku_status.head())


# STEP 4.2 – Lifecycle-Based Action Mapping

def lifecycle_action(stage):
    if stage == 'Introduction':
        return 'Monitor adoption, expand distribution, avoid heavy discounting'
    elif stage == 'Growth':
        return 'Increase marketing, ensure supply readiness, optimize pricing'
    elif stage == 'Maturity':
        return 'Focus on margin optimization, loyalty programs, reduce promos'
    elif stage == 'Decline':
        return 'Investigate causes, reposition or plan phased exit'
    else:
        return 'No action'

latest_sku_status['lifecycle_action'] = (
    latest_sku_status['lifecycle_stage'].apply(lifecycle_action)
)


# STEP 4.3 – Risk-Based Priority Mapping

def risk_priority(risk_level):
    if risk_level == 'High Risk':
        return 'Immediate'
    elif risk_level == 'Medium Risk':
        return 'High'
    else:
        return 'Normal'

latest_sku_status['action_priority'] = (
    latest_sku_status['decline_risk_level'].apply(risk_priority)
)


# STEP 4.4 – Final Business Recommendation Table

business_actions = latest_sku_status[[
    'sku_id',
    'sku_display_name',
    'lifecycle_stage',
    'decline_risk_level',
    'decline_risk_score',
    'action_priority',
    'lifecycle_action'
]].sort_values(
    ['action_priority', 'decline_risk_score'],
    ascending=[True, False]
)

print(business_actions.head(10))


# STEP 4.5 – Priority SKU List (Management Focus)

priority_skus = business_actions[
    (business_actions['lifecycle_stage'] == 'Decline') |
    (
        (business_actions['lifecycle_stage'] == 'Maturity') &
        (business_actions['decline_risk_level'] == 'Medium Risk')
    )
]

print(priority_skus)



# STEP 4.6 – Portfolio Summary Table

portfolio_summary = (
    latest_sku_status
    .groupby(['lifecycle_stage', 'decline_risk_level'])
    .size()
    .reset_index(name='sku_count')
)

print(portfolio_summary)


# STEP 4.7 – Export Step 4 Tables

business_actions.to_csv(
    'business_actions.csv',
    index=False)

priority_skus.to_csv(
    "priority_skus.csv",
    index=False)

portfolio_summary.to_csv(
    "portfolio_summary.csv",
    index=False)

#ML Related
# STEP 4.0 – Ensure proper time ordering (VERY IMPORTANT)

sku_monthly = sku_monthly.sort_values(
    ['sku_id', 'year_month']
).reset_index(drop=True)


# STEP 4.1 – Create target variable
# NOTE: Target represents FUTURE rule-based decline signals
# Used as an EARLY-WARNING proxy, not true market decline

sku_monthly['will_decline_next_3m'] = 0


for sku, grp in sku_monthly.groupby('sku_id'):
    grp = grp.sort_values('year_month')
    
    for i in range(len(grp) - 3):
        future_stages = grp.iloc[i+1:i+4]['lifecycle_stage']
        if 'Decline' in future_stages.values:
            sku_monthly.loc[grp.index[i], 'will_decline_next_3m'] = 1



# STEP 4.2 – Feature list for ML (actual engineered features only)

feature_cols = [
    'product_age_months',
    'sales_growth_pct',
    'rolling_avg_sales_3m',
    'drop_from_peak_pct',
    'consecutive_negative_growth',
    'coefficient_of_variation',
    'promo_frequency',
    'avg_discount'
]


# STEP 4.3 – Prepare ML dataset

ml_data = sku_monthly[feature_cols + ['will_decline_next_3m']].copy()

# Handle missing values safely
ml_data = ml_data.fillna(0)

X = ml_data[feature_cols]
y = ml_data['will_decline_next_3m']


# STEP 4.4 – Time-based train-test split (80/20)

split_idx = int(len(ml_data) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# STEP 4.5 – Logistic Regression (Explainable Baseline)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Performance")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))


# STEP 4.6 – Random Forest Model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Performance")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))


# STEP 4.7 – Feature Importance (Random Forest)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feature_importance)


# STEP 4.8 – ML Decline Probability for ALL rows

sku_monthly['ml_decline_probability'] = rf_model.predict_proba(X)[:, 1]


# STEP 4.9 – ML Risk Levels

def ml_risk_bucket(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

sku_monthly['ml_decline_risk_level'] = (
    sku_monthly['ml_decline_probability'].apply(ml_risk_bucket)
)


# STEP 4.10 – Hybrid Early Warning Flag

sku_monthly['hybrid_early_warning'] = (
    (sku_monthly['ml_decline_risk_level'] == 'High Risk') |
    (
        (sku_monthly['ml_decline_risk_level'] == 'Medium Risk') &
        (sku_monthly['lifecycle_stage'].isin(['Maturity', 'Decline']))
    )
)


# STEP 4.11 – Latest ML Snapshot per SKU

latest_ml_status = (
    sku_monthly
    .sort_values(['sku_id', 'year_month'])
    .groupby('sku_id')
    .tail(1)
)

print(
    latest_ml_status[['sku_display_name',
                      'lifecycle_stage',
                      'ml_decline_probability',
                      'ml_decline_risk_level',
                      'hybrid_early_warning']]
    .sort_values('ml_decline_probability', ascending=False)
)


# STEP 4.12 – Export ML Results

sku_monthly.to_csv(
    "ml_sku_monthly.csv",
    index=False
)

latest_ml_status.to_csv(
    "ml_latest_sku_status.csv",
    index=False
)

feature_importance.to_csv(
    "ml_feature_importance.csv",
    index=False
)


#STEP 5A – Monitoring & Executive Decision Framework

# STEP 5.1 – SKU Health KPIs (Latest Snapshot)

sku_health_kpis = latest_sku_status.copy()

# Revenue contribution (% of total net sales)
total_revenue = sku_health_kpis['total_net_sales'].sum()

sku_health_kpis['revenue_contribution_pct'] = (
    sku_health_kpis['total_net_sales'] / total_revenue * 100
)

# Sales momentum label
def sales_momentum(row):
    if row['decline_risk_score'] < 0.3:
        return 'Healthy'
    elif row['decline_risk_score'] < 0.6:
        return 'Watch'
    else:
        return 'Critical'

sku_health_kpis['sales_momentum'] = sku_health_kpis.apply(
    sales_momentum, axis=1
)

print(sku_health_kpis.head())



# STEP 5.2 – Portfolio Health Summary

portfolio_health = (
    sku_health_kpis
    .groupby(['lifecycle_stage', 'sales_momentum'])
    .agg(
        sku_count=('sku_id', 'count'),
        total_revenue=('total_net_sales', 'sum')
    )
    .reset_index()
)

print(portfolio_health)


# STEP 5.3 – Monthly Decline Monitoring Table

decline_monitoring = sku_monthly[[
    'sku_id',
    'sku_display_name',
    'year_month',
    'lifecycle_stage',
    'decline_risk_score',
    'decline_risk_level'
]].sort_values(['sku_id', 'year_month'])
decline_monitoring.groupby('sku_id').tail(1)


print(decline_monitoring.head())


# STEP 5.4 – Early Warning Flags

sku_health_kpis['early_warning_flag'] = (
    (sku_health_kpis['lifecycle_stage'] == 'Decline') |
    (
        (sku_health_kpis['lifecycle_stage'] == 'Maturity') &
        (sku_health_kpis['decline_risk_level'] == 'Medium Risk')
    )
)
# STEP 5.4.1 – Priority Tag for Executive View

sku_health_kpis['priority_tag'] = sku_health_kpis['early_warning_flag'].apply(
    lambda x: 'PRIORITY SKU' if x else 'Normal SKU'
)


print(
    sku_health_kpis[['sku_display_name',
                     'lifecycle_stage',
                     'decline_risk_level',
                     'early_warning_flag']]
)


# STEP 5.5 – Executive Action Matrix

executive_action_matrix = sku_health_kpis[[
    'sku_id',
    'sku_display_name',
    'lifecycle_stage',
    'sales_momentum',
    'decline_risk_score',
    'decline_risk_level',
    'priority_tag',
    'lifecycle_action'
]].sort_values(
    ['priority_tag', 'decline_risk_score'],
    ascending=[True, False]
)


print(executive_action_matrix.head(10))


#Priority SKU List

priority_skus = executive_action_matrix[
    executive_action_matrix['priority_tag'] == 'PRIORITY SKU'
]

# STEP 5.6 – Export Step 5 Monitoring Tables

sku_health_kpis.to_csv(
    "sku_health_kpis.csv",
    index=False
)

portfolio_health.to_csv(
    "portfolio_health.csv",
    index=False
)

decline_monitoring.to_csv(
    "decline_monitoring.csv",
    index=False
)

executive_action_matrix.to_csv(
    "executive_action_matrix.csv",
    index=False
)

priority_skus.to_csv(
    "priority_skus.csv",
    index=False)


#STEP 5B – Pricing & Promotion Impact Analysis

pricing_df = sku_monthly.copy()
pricing_df = pricing_df.sort_values(['sku_id', 'year_month'])


#STEP 5B.1 — Define High Discount Flag (SKU-relative)
pricing_df['high_discount_flag'] = (
    pricing_df['avg_discount'] >
    pricing_df.groupby('sku_id')['avg_discount'].transform('median')
)


#STEP 5B.2 — Discount Dependency (Core Metric)
discount_dependency = (
    pricing_df
    .groupby('sku_id')
    .agg(
        total_units=('total_units', 'sum'),
        discounted_units=(
            'total_units',
            lambda x: x[pricing_df.loc[x.index, 'high_discount_flag']].sum()
        )
    )
    .reset_index()
)

discount_dependency['discount_dependency_ratio'] = (
    discount_dependency['discounted_units'] /
    discount_dependency['total_units']
).fillna(0)



#STEP 5B.3 — Promotion Intensity
promo_intensity = (
    pricing_df
    .groupby('sku_id')
    .agg(
        avg_promo_frequency=('promo_frequency', 'mean')
    )
    .reset_index()
)


#STEP 5B.4 — Margin Pressure Indicator
margin_pressure = (
    pricing_df
    .groupby('sku_id')
    .agg(
        avg_discount=('avg_discount', 'mean'),
        sales_volatility=('coefficient_of_variation', 'mean')
    )
    .reset_index()
)

margin_pressure['margin_pressure_flag'] = (
    (margin_pressure['avg_discount'] > 0.10) &
    (margin_pressure['sales_volatility'] >
     margin_pressure['sales_volatility'].median())
)



#STEP 5B.5 — Post-Promotion Drop Detection
pricing_df['post_promo_drop_flag'] = (
    (pricing_df['high_discount_flag'].shift(1) == True) &
    (pricing_df['high_discount_flag'] == False) &
    (pricing_df['sales_growth_pct'] < 0)
)

post_promo_drop = (
    pricing_df
    .groupby('sku_id')['post_promo_drop_flag']
    .mean()
    .reset_index(name='post_promo_drop_ratio')
)



#STEP 5B.6 — Price Sensitivity (Elasticity Proxy)
price_sensitivity = (
    pricing_df
    .groupby('sku_id')
    .apply(
        lambda x: (
            x['total_units'].pct_change()
            .corr(x['avg_discount'].pct_change())
        )
    )
    .reset_index(name='price_discount_elasticity')
    .fillna(0)
)



#STEP 5B.7 — Combine Pricing Signals
pricing_risk = (
    discount_dependency
    .merge(promo_intensity, on='sku_id', how='left')
    .merge(margin_pressure, on='sku_id', how='left')
    .merge(post_promo_drop, on='sku_id', how='left')
    .merge(price_sensitivity, on='sku_id', how='left')
)



#STEP 5B.8 — Merge ML Early-Warning Signals
pricing_risk = pricing_risk.merge(
    latest_ml_status[[
        'sku_id',
        'ml_decline_probability',
        'hybrid_early_warning',
        'lifecycle_stage'
    ]],
    on='sku_id',
    how='left'
)



#STEP 5B.9 — Pricing Risk Flag
pricing_risk['pricing_risk_flag'] = (
    (pricing_risk['discount_dependency_ratio'] > 0.4) |
    (pricing_risk['price_discount_elasticity'] > 0.5) |
    (pricing_risk['margin_pressure_flag']) |
    (pricing_risk['post_promo_drop_ratio'] > 0.3)
)



#STEP 5B.10 — Root Cause Classification (🔥 KEY INSIGHT)
def pricing_root_cause(row):
    if row['pricing_risk_flag'] and row['hybrid_early_warning']:
        return 'Structural Decline'
    elif row['pricing_risk_flag'] and not row['hybrid_early_warning']:
        return 'Discount Dependency'
    elif not row['pricing_risk_flag'] and row['hybrid_early_warning']:
        return 'Demand Erosion'
    else:
        return 'Healthy'

pricing_risk['decline_root_cause'] = pricing_risk.apply(
    pricing_root_cause, axis=1
)



#STEP 5B.11 — Final Pricing Impact Table
pricing_impact_summary = pricing_risk[[
    'sku_id',
    'lifecycle_stage',
    'discount_dependency_ratio',
    'avg_promo_frequency',
    'avg_discount',
    'price_discount_elasticity',
    'post_promo_drop_ratio',
    'pricing_risk_flag',
    'ml_decline_probability',
    'hybrid_early_warning',
    'decline_root_cause'
]].sort_values(
    ['pricing_risk_flag', 'ml_decline_probability'],
    ascending=[False, False]
)

print(pricing_impact_summary.head(10))


#STEP 5B.12 — Export
pricing_impact_summary.to_csv(
    "pricing_impact_summary.csv",
    index=False
)


# STEP 6.1 – Select SKUs for Forecasting

forecast_skus = latest_ml_status[
    (latest_ml_status['hybrid_early_warning'] == True) |
    (latest_ml_status['ml_decline_probability'] >= 0.4)
]['sku_id'].unique()

forecast_df = sku_monthly[
    sku_monthly['sku_id'].isin(forecast_skus)
].copy()



# STEP 6.2 – Create numeric time index per SKU
forecast_df['time_index'] = (
    forecast_df
    .groupby('sku_id')
    .cumcount()
)


#STEP 6.3 – Baseline Forecast (Linear Trend)
from sklearn.linear_model import LinearRegression

forecast_results = []

for sku, grp in forecast_df.groupby('sku_id'):
    grp = grp.sort_values('year_month')

    if len(grp) < 6:
        continue  # insufficient history

    X = grp[['time_index']]
    y = grp['total_net_sales']

    model = LinearRegression()
    model.fit(X, y)

    # Forecast next 3 months
    last_index = grp['time_index'].max()
    future_index = [[last_index + i] for i in range(1, 4)]

    base_forecast = model.predict(future_index)

    forecast_results.append({
        'sku_id': sku,
        'base_forecast_3m_avg': base_forecast.mean()
    })



#STEP 6.4 – Risk-Adjusted Forecast
forecast_outlook = pd.DataFrame(forecast_results)

forecast_outlook = forecast_outlook.merge(
    latest_ml_status[['sku_id', 'ml_decline_probability']],
    on='sku_id',
    how='left'
)

forecast_outlook['risk_adjusted_forecast'] = (
    forecast_outlook['base_forecast_3m_avg'] *
    (1 - forecast_outlook['ml_decline_probability'])
)

forecast_outlook['downside_risk_pct'] = (
    (forecast_outlook['base_forecast_3m_avg'] -
     forecast_outlook['risk_adjusted_forecast']) /
    forecast_outlook['base_forecast_3m_avg'] * 100
)

forecast_outlook['revenue_at_risk'] = (
    forecast_outlook['base_forecast_3m_avg'] -
    forecast_outlook['risk_adjusted_forecast']
)



#STEP 6.5 – Final Step 6 Output
forecast_outlook = forecast_outlook.sort_values(
    'downside_risk_pct',
    ascending=False
)

print(forecast_outlook.head(10))


# STEP 6.6 – Scenario Band Forecasts
#STEP 6.6.1 — Scenario Forecast Calculations
# Best Case (risk partially materializes)
forecast_outlook['best_case_forecast'] = (
    forecast_outlook['base_forecast_3m_avg'] *
    (1 - 0.5 * forecast_outlook['ml_decline_probability'])
)

# Base Case (already computed, but keep for clarity)
forecast_outlook['base_case_forecast'] = (
    forecast_outlook['risk_adjusted_forecast']
)

# Worst Case (risk over-materializes, capped at 100%)
forecast_outlook['worst_case_forecast'] = (
    forecast_outlook['base_forecast_3m_avg'] *
    (1 - (1.2 * forecast_outlook['ml_decline_probability']).clip(upper=1))
)


#STEP 6.6.2 — Scenario Revenue at Risk
forecast_outlook['best_case_revenue_loss'] = (
    forecast_outlook['base_forecast_3m_avg'] -
    forecast_outlook['best_case_forecast']
)

forecast_outlook['worst_case_revenue_loss'] = (
    forecast_outlook['base_forecast_3m_avg'] -
    forecast_outlook['worst_case_forecast']
)


#STEP 6.6.3 — Final Scenario View (Executive Table)
scenario_outlook = forecast_outlook[[
    'sku_id',
    'base_forecast_3m_avg',
    'best_case_forecast',
    'base_case_forecast',
    'worst_case_forecast',
    'best_case_revenue_loss',
    'revenue_at_risk',
    'worst_case_revenue_loss'
]].sort_values(
    'revenue_at_risk',
    ascending=False
)

print(scenario_outlook.head(10))


transactions=pd.read_csv('synthetic_fmcg_transactions.csv')
transactions.columns


from mlxtend.frequent_patterns import fpgrowth, association_rules

# =========================================================
# STEP 7 — MARKET BASKET ANALYSIS (MBA)
# Purpose:
#   • Understand what customers buy ALONG WITH declining / risky SKUs
#   • Use transaction-level data (~6M rows)
#   • Feed actionable insights into Step 8 (Bundles & Combos)
#
# IMPORTANT DESIGN PRINCIPLE:
#   Declining SKUs are ANCHORS, not the entire basket.
#   We analyze transactions that CONTAIN risky SKUs,
#   but allow ALL SKUs to participate in the basket.
# =========================================================


# ---------------------------------------------------------
# STEP 7.2 — Define Focus SKUs (FROM STEP 4 / STEP 6 OUTPUT)
# ---------------------------------------------------------
# Focus SKUs are those flagged by the hybrid early-warning system
# This ensures full alignment with Steps 4–6

focus_skus = set(
    latest_ml_status.loc[
        latest_ml_status['hybrid_early_warning'] == True,
        'sku_id'
    ]
)

print("Number of focus (risky) SKUs:", len(focus_skus))


# ---------------------------------------------------------
# STEP 7.3 — Select Transactions That Contain Risky SKUs
# ---------------------------------------------------------
# We DO NOT filter SKUs here.
# We filter TRANSACTIONS that contain at least one risky SKU.

risk_txn_ids = transactions.loc[
    transactions['sku_id'].isin(focus_skus),
    'transaction_id'
].unique()

txn_focus = transactions[
    transactions['transaction_id'].isin(risk_txn_ids)
].copy()

print("Transactions containing risky SKUs:", txn_focus.shape)


# ---------------------------------------------------------
# STEP 7.4 — Create Transaction Baskets
# ---------------------------------------------------------
# Basket definition:
#   • transaction_id = basket
#   • sku_id = item
#   • quantity > 0 → presence (True/False)
#
# Boolean baskets are REQUIRED for FP-Growth performance.

basket = (
    txn_focus
    .groupby(['transaction_id', 'sku_id'])['quantity']
    .sum()
    .unstack(fill_value=0)
)

# Convert to boolean (fixes performance warning)
basket = basket > 0

print("Initial basket shape:", basket.shape)


# ---------------------------------------------------------
# STEP 7.5 — Prune Rare SKUs (Dimensionality Reduction)
# ---------------------------------------------------------
# Remove SKUs that appear in fewer than 0.2% of baskets
# This prevents noisy and meaningless rules.

min_support_txn = 0.002 * len(basket)

basket = basket.loc[:, basket.sum() >= min_support_txn]

print("Basket after pruning:", basket.shape)


# ---------------------------------------------------------
# STEP 7.6 — Frequent Itemset Mining (FP-Growth)
# ---------------------------------------------------------
# FP-Growth is chosen over Apriori for scalability.

frequent_itemsets = fpgrowth(
    basket,
    min_support=0.005,   # 0.5% support
    use_colnames=True
)

print("Frequent itemsets found:", frequent_itemsets.shape)
print(frequent_itemsets.head())


# ---------------------------------------------------------
# STEP 7.7 — Association Rule Generation
# ---------------------------------------------------------
# We prioritize confidence first, then lift.

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3
)

rules = rules.sort_values(
    ['lift', 'confidence'],
    ascending=False
)

print("Total association rules:", rules.shape)
print(rules.head())


# ---------------------------------------------------------
# STEP 7.8 — Keep Only Rules Anchored on Declining SKUs
# ---------------------------------------------------------
# Business logic:
#   • Antecedent must contain at least one risky SKU
#   • Consequent can be ANY SKU (cross-sell opportunity)

rules['antecedents_sku'] = rules['antecedents'].apply(list)
rules['consequents_sku'] = rules['consequents'].apply(list)

decline_rules = rules[
    rules['antecedents_sku'].apply(
        lambda x: any(sku in focus_skus for sku in x)
    )
]

print("Decline-anchored rules:", decline_rules.shape)
print(decline_rules.head())


# ---------------------------------------------------------
# STEP 7.9 — Final Business-Friendly Output Table
# ---------------------------------------------------------
mba_output = decline_rules[[
    'antecedents_sku',
    'consequents_sku',
    'support',
    'confidence',
    'lift'
]].reset_index(drop=True)

print("Final MBA output:")
print(mba_output.head(10))


# ---------------------------------------------------------
# STEP 7.10 — Export Step 7 Results
# ---------------------------------------------------------
mba_output.to_csv(
    "mba_rules_declining_skus.csv",
    index=False
)


# =========================================================
# STEP 8 — COMBO / BUNDLE RECOMMENDATION SYSTEM
# =========================================================

import ast


# ---------------------------------------------------------
# STEP 8.1 — Load Inputs
# ---------------------------------------------------------

mba_rules = pd.read_csv(
    "mba_rules_declining_skus.csv"
)

latest_ml_status = latest_ml_status.copy()  # already in memory


# ---------------------------------------------------------
# STEP 8.2 — Parse SKU Lists
# ---------------------------------------------------------

mba_rules['antecedents_sku'] = mba_rules['antecedents_sku'].apply(ast.literal_eval)
mba_rules['consequents_sku'] = mba_rules['consequents_sku'].apply(ast.literal_eval)


# ---------------------------------------------------------
# STEP 8.3 — Identify Anchor (Declining) SKUs
# ---------------------------------------------------------

focus_skus = set(
    latest_ml_status.loc[
        latest_ml_status['hybrid_early_warning'] == True,
        'sku_id'
    ]
)


# ---------------------------------------------------------
# STEP 8.4 — Explode Rules into Bundle Candidates
# ---------------------------------------------------------

bundle_rows = []

for _, row in mba_rules.iterrows():
    anchor_skus = [sku for sku in row['antecedents_sku'] if sku in focus_skus]
    attach_skus = row['consequents_sku']

    for anchor in anchor_skus:
        for attach in attach_skus:
            bundle_rows.append({
                'anchor_sku': anchor,
                'bundle_sku': attach,
                'support': row['support'],
                'confidence': row['confidence'],
                'lift': row['lift']
            })

bundle_df = pd.DataFrame(bundle_rows)

print("Total bundle candidates:", bundle_df.shape)


# ---------------------------------------------------------
# STEP 8.5 — Enrich with Risk & Lifecycle Context
# ---------------------------------------------------------

bundle_df = bundle_df.merge(
    latest_ml_status[['sku_id', 'lifecycle_stage', 'ml_decline_probability']],
    left_on='anchor_sku',
    right_on='sku_id',
    how='left'
).drop(columns=['sku_id'])


# ---------------------------------------------------------
# STEP 8.6 — Bundle Strength Score
# ---------------------------------------------------------
# Weighted score emphasizing business confidence

bundle_df['bundle_score'] = (
    0.5 * bundle_df['confidence'] +
    0.3 * (bundle_df['lift'] / bundle_df['lift'].max()) +
    0.2 * bundle_df['support']
)


# ---------------------------------------------------------
# STEP 8.7 — Rank & Select Top Bundles per Anchor SKU
# ---------------------------------------------------------

# Aggregate duplicate bundle pairs
bundle_df_agg = (
    bundle_df
    .groupby(['anchor_sku', 'bundle_sku'], as_index=False)
    .agg(
        support=('support', 'max'),
        confidence=('confidence', 'max'),
        lift=('lift', 'max'),
        ml_decline_probability=('ml_decline_probability', 'first'),
        lifecycle_stage=('lifecycle_stage', 'first')
    )
)

# Recompute bundle score after aggregation
bundle_df_agg['bundle_score'] = (
    0.5 * bundle_df_agg['confidence'] +
    0.3 * (bundle_df_agg['lift'] / bundle_df_agg['lift'].max()) +
    0.2 * bundle_df_agg['support']
)


#Select top bundles per anchor (FINAL)
top_bundles = (
    bundle_df_agg
    .sort_values(['anchor_sku', 'bundle_score'], ascending=[True, False])
    .groupby('anchor_sku')
    .head(3)   # keep top 3 bundles per SKU
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# STEP 8.8 — Business-Friendly Recommendation Text
# ---------------------------------------------------------

def bundle_action(row):
    if row['ml_decline_probability'] >= 0.6:
        return "Bundle to arrest decline and preserve volume"
    elif row['ml_decline_probability'] >= 0.4:
        return "Bundle to stabilize demand"
    else:
        return "Optional cross-sell bundle"


top_bundles['recommended_action'] = top_bundles.apply(bundle_action, axis=1)


# ---------------------------------------------------------
# STEP 8.9 — Final Executive Output
# ---------------------------------------------------------

bundle_recommendations = top_bundles[[
    'anchor_sku',
    'bundle_sku',
    'lifecycle_stage',
    'ml_decline_probability',
    'confidence',
    'lift',
    'bundle_score',
    'recommended_action'
]].sort_values(
    ['ml_decline_probability', 'bundle_score'],
    ascending=False
)

print(bundle_recommendations.head(10))


# ---------------------------------------------------------
# STEP 8.10 — Export Results
# ---------------------------------------------------------

bundle_recommendations.to_csv(
    "bundle_recommendations.csv",
    index=False
)



