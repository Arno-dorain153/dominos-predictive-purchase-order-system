import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sales_path = os.path.join(base_dir, 'Pizza_Sale - pizza_sales.csv')
    ingr_path = os.path.join(base_dir, 'Pizza_ingredients - Pizza_ingredients.csv')
    
    print("\n=========================================================================")
    print("      STEP 1: ENHANCED PREPROCESSING, EDA & CORRELATION ANALYSIS        ")
    print("=========================================================================")
    
    # -------------------------------------------------------------------------
    # 1. SALES DATASET AUDIT & CLEANING
    # -------------------------------------------------------------------------
    print(f"\n[FUNCTION 1]: Processing Raw Sales Dataset")
    print(f"  -> Path: {sales_path}")
    sales_df = pd.read_csv(sales_path)
    initial_sales_rows = len(sales_df)
    print(f"  -> Total Raw Records Loaded: {initial_sales_rows} transaction rows")
    
    # Check Duplicate Data
    sales_duplicates = sales_df.duplicated().sum()
    print(f"\n  [Audit A] Duplicate Data Check:")
    print(f"    * Duplicate Rows Detected: {sales_duplicates}")
    if sales_duplicates > 0:
        sales_df.drop_duplicates(inplace=True)
        print(f"    [OK] Dropped {sales_duplicates} duplicate rows. Remaining: {len(sales_df)}")
    else:
        print("    [OK] No duplicate rows found in sales dataset.")
        
    # Check Missing / Empty Data
    missing_before = sales_df.isnull().sum()
    total_missing_sales = missing_before.sum()
    print(f"\n  [Audit B] Missing / Empty Data Check:")
    for col, count in missing_before.items():
        if count > 0:
            print(f"    - Column '{col}': {count} missing values")
    if total_missing_sales == 0:
        print("    - All columns are 100% complete (0 missing values).")
        
    # Standardize & parse order_date
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], dayfirst=True, errors='coerce', format='mixed')
    null_dates = sales_df['order_date'].isnull().sum()
    if null_dates > 0:
        sales_df.dropna(subset=['order_date'], inplace=True)
        print(f"    [OK] Dropped {null_dates} invalid date rows (0.04% of data).")
    
    # Clean ingredient text
    sales_df['pizza_ingredients'] = sales_df['pizza_ingredients'].astype(str).str.replace(r'[^a-zA-Z\s,]', '', regex=True)
    
    # Feature Engineering (Calendar, Cyclic & Holiday Features)
    print(f"\n  [Audit C] Advanced Feature Engineering:")
    sales_df['day_of_week'] = sales_df['order_date'].dt.dayofweek
    sales_df['month'] = sales_df['order_date'].dt.month
    sales_df['year'] = sales_df['order_date'].dt.year
    sales_df['day'] = sales_df['order_date'].dt.day
    sales_df['day_of_year'] = sales_df['order_date'].dt.dayofyear
    
    # Cyclical Date Encoding (Sin / Cos Transformations)
    sales_df['sin_day'] = np.sin(2 * np.pi * sales_df['day_of_year'] / 365.25)
    sales_df['cos_day'] = np.cos(2 * np.pi * sales_df['day_of_year'] / 365.25)
    sales_df['is_weekend'] = np.where(sales_df['day_of_week'].isin([4, 5, 6]), 1, 0)
    
    # Major US Holidays
    holidays = pd.to_datetime(['2015-01-01', '2015-07-04', '2015-11-26', '2015-12-25'])
    sales_df['is_holiday'] = np.where(sales_df['order_date'].isin(holidays), 1, 0)
    print("    [OK] Added cyclic features ('sin_day', 'cos_day'), 'is_weekend', and 'is_holiday'.")
    
    # IQR Outlier Cleaning on Quantity
    print(f"\n  [Audit D] Outlier Removal (IQR 1.5x Method):")
    Q1 = sales_df['quantity'].quantile(0.25)
    Q3 = sales_df['quantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_sales = sales_df[(sales_df['quantity'] >= lower_bound) & (sales_df['quantity'] <= upper_bound)].copy()
    outliers_removed = len(sales_df) - len(cleaned_sales)
    print(f"    * Quantity IQR Range: [{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"    * Removed {outliers_removed} extreme volume spike rows.")
    print(f"    [OK] Cleaned Sales Dataset Final Rows: {len(cleaned_sales)}")
    
    output_sales_path = os.path.join(base_dir, 'cleaned_pizza2.csv')
    cleaned_sales.to_csv(output_sales_path, index=False)
    print(f"  -> Exported cleaned sales file to: {output_sales_path}")

    # -------------------------------------------------------------------------
    # 2. EDA BAR GRAPHS & CORRELATION MATRIX HEATMAP
    # -------------------------------------------------------------------------
    print(f"\n[FUNCTION 2]: Generating EDA Bar Graphs & Correlation Matrix...")
    
    # EDA Bar Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cat_counts = cleaned_sales['pizza_category'].value_counts()
    axes[0].bar(cat_counts.index.astype(str), cat_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_title('Count of Sales by Pizza Category', fontsize=14)
    axes[0].set_xlabel('Pizza Category', fontsize=12)
    axes[0].set_ylabel('Total Orders', fontsize=12)
    axes[0].tick_params(axis='x', rotation=30)
    for i, v in enumerate(cat_counts.values):
        axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    size_counts = cleaned_sales['pizza_size'].value_counts()
    axes[1].bar(size_counts.index.astype(str), size_counts.values, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    axes[1].set_title('Count of Sales by Pizza Size', fontsize=14)
    axes[1].set_xlabel('Pizza Size', fontsize=12)
    axes[1].set_ylabel('Total Orders', fontsize=12)
    for i, v in enumerate(size_counts.values):
        axes[1].text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    eda_path = os.path.join(base_dir, 'eda_charts.png')
    plt.savefig(eda_path, dpi=300)
    plt.close()
    print(f"  -> Saved EDA Bar Graphs plot to: {eda_path}")

    # Correlation Matrix Heatmap
    corr_cols = ['quantity', 'unit_price', 'total_price', 'day_of_week', 'month', 'day', 'is_holiday', 'is_weekend']
    corr_matrix = cleaned_sales[corr_cols].corr()
    
    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im)
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right')
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title('Correlation Matrix Heatmap', fontsize=14)
    
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr_matrix.iloc[i, j]
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if abs(val) > 0.5 else 'black', fontsize=9)
            
    plt.tight_layout()
    corr_path = os.path.join(base_dir, 'correlation_matrix.png')
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"  -> Saved Correlation Matrix Heatmap to: {corr_path}")

    # -------------------------------------------------------------------------
    # 3. INGREDIENTS DATASET AUDIT & CLEANING
    # -------------------------------------------------------------------------
    print(f"\n[FUNCTION 3]: Processing Ingredient Recipe Dataset")
    print(f"  -> Path: {ingr_path}")
    ingr_df = pd.read_csv(ingr_path)
    initial_ingr_rows = len(ingr_df)
    print(f"  -> Total Raw Recipe Records Loaded: {initial_ingr_rows} ingredient mapping rows")
    
    ingr_duplicates = ingr_df.duplicated().sum()
    if ingr_duplicates > 0:
        ingr_df.drop_duplicates(inplace=True)
        
    ingr_missing = ingr_df.isnull().sum()
    if ingr_missing.sum() > 0:
        ingr_df.dropna(inplace=True)
        
    ingr_df['pizza_ingredients'] = ingr_df['pizza_ingredients'].astype(str).str.lower().str.strip()
    
    Q1_ing = ingr_df['Items_Qty_In_Grams'].quantile(0.25)
    Q3_ing = ingr_df['Items_Qty_In_Grams'].quantile(0.75)
    IQR_ing = Q3_ing - Q1_ing
    lower_ing = Q1_ing - 1.5 * IQR_ing
    upper_ing = Q3_ing + 1.5 * IQR_ing
    cleaned_ingr = ingr_df[(ingr_df['Items_Qty_In_Grams'] >= lower_ing) & (ingr_df['Items_Qty_In_Grams'] <= upper_ing)].copy()
    
    output_ingr_path = os.path.join(base_dir, 'cleaned_ingredients.csv')
    cleaned_ingr.to_csv(output_ingr_path, index=False)
    print(f"  -> Exported cleaned ingredient file to: {output_ingr_path}")
    print("=========================================================================\n")
    
    return cleaned_sales, cleaned_ingr

if __name__ == '__main__':
    preprocess_data()
