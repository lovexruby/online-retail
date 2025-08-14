import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('online_retail.csv')

"""
# Check Dataframe
print(df.info())

# Check for null values and duplicates
print(df.isnull().sum())
print("Duplicates: ", df.duplicated().sum())
"""

# Calculating Revenue
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Convert Date into Datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Filter Nan CustomerIDs
df_customers = df.dropna(subset=['CustomerID']).copy()
df_customers['CustomerID'] = df_customers['CustomerID'].astype(int)

# Calculate revenue per customer
top_customers = df_customers.groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False)
print(top_customers.head(10))

# Calculate revenue per country
revenue_country = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
print(revenue_country.head(10))

# Calculate revenue per timeslot
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
revenue_month = df.groupby('InvoiceMonth')['Revenue'].sum()
print(revenue_month)

# Filter out negative Quantity (Retour)
df_positive = df[df['Quantity'] > 0]

# Products per Invoice
basket = df_positive.groupby('InvoiceNo')['StockCode'].apply(list)
print(basket.head())

revenue_month.plot(kind='line', figsize=(10,6), title='Revenue per month')
plt.ylabel('Revenue')
plt.xlabel('Month')
plt.show()