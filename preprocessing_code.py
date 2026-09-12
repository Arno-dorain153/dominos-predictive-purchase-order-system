# --- CELL 0 (code) ---
import pandas as pd

# Load the dataset (you can specify the path to your file)
df = pd.read_csv('C:/Users/Siva/Downloads/Pizza_Sale - pizza_sales.csv')

df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True, errors='coerce', format='mixed')
df.dropna()
df.info()
df.dropna()

# --- CELL 1 (code) ---
# Check for duplicates
duplicates = df.duplicated().sum()
print(f'Duplicates: {duplicates}')


# --- CELL 2 (code) ---
import pandas as pd

# Remove special characters
df['pizza_ingredients'] = df['pizza_ingredients'].str.replace(r'[^a-zA-Z\s]', '', regex=True)




# --- CELL 3 (code) ---
# Convert order_date and order_time to datetime
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_time'] = pd.to_datetime(df['order_time'], format='%H:%M:%S').dt.time


# --- CELL 4 (code) ---
import seaborn as sns
import matplotlib.pyplot as plt

# Plot histograms
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['quantity'], bins=30, kde=True)
plt.title('Quantity Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['unit_price'], bins=30, kde=True)
plt.title('Unit Price Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['total_price'], bins=30, kde=True)
plt.title('Total Price Distribution')

plt.tight_layout()
plt.show()


# --- CELL 5 (code) ---
# Countplot for pizza categories
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='pizza_category')
plt.title('Count of Pizza Categories')
plt.xticks(rotation=45)
plt.show()

# Countplot for pizza sizes
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='pizza_size')
plt.title('Count of Pizza Sizes')
plt.show()


# --- CELL 6 (code) ---
# Correlation matrix
corr_matrix = df[['quantity', 'unit_price', 'total_price']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# --- CELL 7 (code) ---
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='pizza_category', y='total_price')
plt.title('Total Price by Pizza Category')
plt.xticks(rotation=45)
plt.show()


# --- CELL 8 (code) ---
# Group by order_date and sum total_price
daily_sales = df.groupby('order_date')['total_price'].sum().reset_index()

# Plot daily sales
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['order_date'], daily_sales['total_price'])
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# --- CELL 9 (code) ---
# Extracting relevant date features
df['day_of_week'] = df['order_date'].dt.day_name()  # Day of the week
df['month'] = df['order_date'].dt.month  # Month
df['year'] = df['order_date'].dt.year  # Year
df['day'] = df['order_date'].dt.day  # Day of the month


# --- CELL 10 (code) ---
import numpy as np

# Define a list of holiday dates
holidays = pd.to_datetime(['2015-12-25'])  # Example holidays
df['is_holiday'] = np.where(df['order_date'].isin(holidays), 1, 0)


# --- CELL 11 (code) ---
# One-hot encoding for categorical features
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Encode day_of_week and month
df['day_of_week'] = label_encoder.fit_transform(df['day_of_week'])
df['month'] = label_encoder.fit_transform(df['month'])



# --- CELL 12 (code) ---
# Check the updated DataFrame with new features
print(df.head())


# --- CELL 13 (code) ---
df.to_csv('cleaned_pizza1.csv', index=False)

# --- CELL 14 (code) ---
import pandas as pd

df = pd.read_csv('D:/project 5/Cleaned_Pizza_ingredients.csv')

df.info()
df.head()

# --- CELL 15 (code) ---
# Check for duplicate entries
duplicates = df.duplicated(subset=['pizza_name_id', 'pizza_name'], keep=False)
print(f"Number of duplicates: {duplicates.sum()}")


# --- CELL 16 (code) ---
# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# If there were any missing values, you could fill them or drop them
# df_cleaned = df_cleaned.dropna()  # To drop
# df_cleaned = df_cleaned.fillna(value)  # To fill


# --- CELL 17 (code) ---
# Standardizing ingredient names to lowercase
df['pizza_ingredients'] = df['pizza_ingredients'].str.lower()


# --- CELL 18 (code) ---
df.to_csv('cleaned_ingredients1.csv', index=False)

# --- CELL 19 (code) ---
import pandas as pd

# Assuming the DataFrame is named df
df = pd.read_csv('C:/Users/Siva/cleaned_ingredients1.csv')
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Items_Qty_In_Grams'].quantile(0.25)
Q3 = df['Items_Qty_In_Grams'].quantile(0.75)

# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1

# Define the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame to remove outliers
df_cleaned = df[(df['Items_Qty_In_Grams'] >= lower_bound) & (df['Items_Qty_In_Grams'] <= upper_bound)]

# Display the cleaned DataFrame
print(df_cleaned)


# --- CELL 20 (code) ---
df_cleaned.to_csv('D:/project 5/new/cleaned_ingredients.csv', index=False)

# --- CELL 21 (code) ---
import pandas as pd

# Sample DataFrame creation (replace this with your actual DataFrame)
data = pd.read_csv('C:/Users/Siva/cleaned_pizza1.csv')

df = pd.DataFrame(data)

# Calculate Q1 and Q3
Q1 = df['quantity'].quantile(0.25)
Q3 = df['quantity'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame to remove outliers
df_cleaned = df[(df['quantity'] >= lower_bound) & (df['quantity'] <= upper_bound)]

# Display the cleaned DataFrame
print(df_cleaned)


# --- CELL 22 (code) ---
df_cleaned.to_csv('D:/project 5/new/cleaned_pizza.csv', index=False)

# --- CELL 23 (code) ---


