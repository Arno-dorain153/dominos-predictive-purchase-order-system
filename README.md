# Dominos Predictive Purchase Order & Inventory System

An end-to-end Machine Learning, Time-Series Forecasting, and Procurement Optimization System for Dominos, paired with an interactive Web Portal with Role-Based Access Control (RBAC).

---

## 💻 How to Run in Terminal

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn statsmodels prophet matplotlib seaborn fastapi uvicorn streamlit
```

### 2. Step 1: Preprocess Datasets & Generate EDA Visualizations
Runs data cleaning, handles missing values, removes outliers (IQR method), extracts calendar features, and generates EDA & Correlation charts:
```bash
python run_preprocessing.py
```

### 3. Step 2: Run Machine Learning Pipeline & Purchase Order Generator
Evaluates 5 forecasting models (SARIMA, ARIMA, Random Forest, ExtraTrees, Polynomial Regression), performs 5-Fold Cross Validation, generates ROC & Confusion Matrix plots, and exports the 7-day ingredient purchase order:
```bash
python run_pipeline.py
```

### 4. Step 3: Launch Interactive Dominos Web Portal (FastAPI)
Launches the Web Application with Dominos branding (Dominos Blue `#006491`, Dominos Red `#E31837`), interactive timeline filters, financial analytics, and purchase order tables:
```bash
python server.py
```
👉 Open your browser and navigate to: **`http://localhost:8000`**

### 5. Step 4 (Optional): Launch Streamlit Dashboard
```bash
streamlit run app.py
```
👉 Open your browser and navigate to: **`http://localhost:8501`**

---

## 🔐 Login Credentials (RBAC Profiles)

| Profile Role | Username | Password | Dedicated View |
|---|---|---|---|
| 👔 **Upper Management (Admin)** | `admin` | `admin123` | Gross Sales Revenue ($786.3k), Food Ingredient Spend (~30%), Food Wastage Cost & ML Savings, Timeline Filter, Lowest Wastage Month Highlight, Supplier PO Approvals. |
| 🍕 **Kitchen Employee** | `employee` | `dominos123` | Upcoming 7-Day Food Purchase Order, Base Weight (kg), Safety Stock Buffer Control (+5% to +25%), Recommended 5kg Supplier Packs, Search & CSV Export. |

---

## 🎯 What to Expect

### 📈 Model Performance Benchmark:
* **Best Forecasting Model**: **SARIMA $(1,0,1)\times(1,0,1)_7$** with **17.10% MAPE** and **30.58 RMSE**.
* **Best Classification Model**: **Tuned ExtraTrees Classifier** with **74.54% Precision** and **0.7335 ROC-AUC**.

### 📊 Generated Image & Data Artifacts:
* `cleaned_pizza2.csv`: Preprocessed daily sales records.
* `cleaned_ingredients.csv`: Standardized ingredient recipe weights.
* `ingredient_purchase_order.csv`: Detailed 7-day purchasing list for all 62 raw ingredients with 10% safety stock.
* `correlation_matrix.png`: Heatmap of sales and calendar feature correlations.
* `confusion_matrix.png`: Confusion matrix heatmap for the top classification model.
* `feature_importance.png`: Relative feature importance bar chart.
* `roc_curve.png`: Multi-model ROC-AUC comparison plot.
* `eda_charts.png`: Breakdown of sales by pizza category and pizza size.
