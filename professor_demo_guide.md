# Professor Demo Guide & Defense Q&A: Dominos Predictive Purchase Order System

This document provides exact, structured answers for your professor's evaluation questions, including **Mandatory Time-Series Cross Validation**, **Classification & Regression Evaluation Metrics**, and the **ROC-AUC Curve**.

---

## 🎯 Question 1: Choose the Objective

### Business Objective
To optimize Dominos' supply chain by eliminating food waste from ingredient expiration (over-ordering) and preventing stockouts/lost revenue (under-ordering).

### Data Science / Machine Learning Objective
1. **Time-Series Demand Forecasting**: Develop a 7-day predictive sales model for total daily pizza demand using 1 year of historical transactional data (2015).
2. **High-Demand / Stockout Risk Classification**: Binary classification to predict whether a given day will experience **High Demand** ($y = 1$) vs **Baseline Demand** ($y = 0$).
3. **Bill-of-Materials (BOM) Conversion**: Map 7-day predicted pizza sales across 91 pizza variants to individual raw ingredient quantities (grams/kilograms).
4. **Automated Purchase Order System**: Generate an actionable, optimized purchase order listing exact ingredient quantities needed for the next week, including a **10% safety stock buffer**.

---

## 📊 Question 2: Identify the Dataset

The system relies on two primary dataset sources:

| Dataset | File Name | Size / Records | Key Attributes | Purpose |
|---|---|---|---|---|
| **Sales Transaction Dataset** | `Pizza_Sale - pizza_sales.csv` | 48,620 transactions | `pizza_id`, `order_id`, `pizza_name_id`, `quantity`, `order_date`, `order_time`, `unit_price`, `total_price`, `pizza_size`, `pizza_category` | Historical demand timeline across 358 unique operating days in 2015. |
| **Ingredient BOM Dataset** | `Pizza_ingredients - Pizza_ingredients.csv` | 518 ingredient rows | `pizza_name_id`, `pizza_name`, `pizza_ingredients`, `Items_Qty_In_Grams` | Mapping of exact grams of each raw ingredient required per pizza variant. |

---

## 🧹 Question 3: Preprocessing & Mandatory Cross-Validation

### Step 1: Data Preprocessing & Cleaning
- **Format Normalization**: Standardized `order_date` using `pd.to_datetime(format='mixed', dayfirst=True)`.
- **Handling Missing Values & Duplicates**: Dropped records with missing dates and removed duplicate transactions.
- **Resampling**: Aggregated 48,620 line items into 358 daily totals using `.groupby('order_date').agg({'quantity': 'sum'})`.
- **Outlier Removal (IQR)**: Filtered extreme sales spikes using $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$.

### Step 2: Mandatory 5-Fold Time-Series Cross Validation (`TimeSeriesSplit`)
To prevent temporal data leakage, we implemented **5-Fold Time-Series Cross Validation** (`TimeSeriesSplit(n_splits=5)`):

| Model | CV MSE | CV RMSE | CV MAE | CV R² Score | CV MAPE (%) |
|---|---|---|---|---|---|
| **Polynomial Regression (Deg 3)** | 5111.20 | 53.22 | 39.62 | -12.18 | 29.59% |
| **Random Forest Regressor (Lags)** | **565.02** | **23.15** | **16.37** | **0.1243** | **12.77%** |

---

## 🤖 Question 4 & 5: Apply Multiple Models & Full Evaluation Metrics

We evaluated both **Regression Forecasting Models** and **Demand Level Classification Models**.

### 1. Regression Model Evaluation (MSE, RMSE, MAE, R², MAPE)

| Model Architecture | MSE | RMSE | MAE | R² Score | MAPE (%) | Selection Status |
|---|---|---|---|---|---|---|
| **SARIMA $(1,0,1)\times(1,0,1)_7$** | **935.08** | **30.58** | **20.99** | **0.0081** | **17.10%** | **BEST FORECAST MODEL** |
| **ARIMA $(1,1,1)$** | 945.74 | 30.75 | 21.15 | -0.0032 | 17.14% | Runner Up |
| **Random Forest Regressor** | 935.86 | 30.59 | 22.03 | 0.0073 | 18.64% | ML Baseline |
| **Polynomial Regression (Deg 3)** | 1220.50 | 34.94 | 26.40 | -0.2946 | 22.80% | Polynomial Baseline |

---

### 2. Demand Classification & ROC / AUC Evaluation
Target: Predict **High Demand / Stockout Risk Day** ($1$ if daily quantity $>$ median, else $0$).

| Classifier Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1 Score | MSE | RMSE | ROC-AUC Score |
|---|---|---|---|---|---|---|---|---|
| **Gradient Boosting Classifier** | **69.01%** | 0.6786 | 0.5938 | 0.7692 | 0.6333 | 0.3099 | 0.5566 | **0.7316** |
| **Random Forest Classifier** | 67.61% | 0.6452 | **0.6250** | 0.7179 | **0.6349** | 0.3239 | 0.5692 | 0.7196 |
| **Logistic Regression** | **70.42%** | **0.7200** | 0.5625 | **0.8205** | 0.6316 | **0.2958** | **0.5439** | 0.6843 |

* **ROC Curve Plot**: Generated and saved to `roc_curve.png`.

---

## 📦 Generated Purchase Order Output (Top 10 Ingredients)

Based on the 7-day SARIMA forecast (total 862.3 pizzas predicted), the exact Bill-of-Materials algorithm produced:

```csv
pizza_ingredients,total_grams,total_kg
red onions,16598.0,16.60
tomatoes,11397.0,11.40
pepperoni,8240.0,8.24
mushrooms,7249.0,7.25
garlic,5967.0,5.97
spinach,5816.0,5.82
mozzarella cheese,4156.0,4.16
red peppers,3455.0,3.46
goat cheese,3008.0,3.01
pineapple,2950.0,2.95
```

---

## 🚀 How to Run the Demo for Your Professor

```bash
# Step 1: Preprocess datasets
python run_preprocessing.py

# Step 2: Run Time-Series CV, Regression Metrics, Classification Metrics (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC), and export Purchase Order
python run_pipeline.py
```
