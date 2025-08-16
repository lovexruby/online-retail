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


# RFM-Analysis
# Filter out customers with Nan CustomerID
df_cust = df.dropna(subset=['CustomerID']).copy()
df_cust['CustomerID'] = df_cust['CustomerID'].astype(int)

# Filter out retours
mask_returns = (
    df_cust['InvoiceNo'].astype(str).str.startswith('C') |
    (df_cust['Quantity'] <= 0) |
    (df_cust['UnitPrice'] <= 0)
)
df_rfm = df_cust.loc[~mask_returns].copy()

# Referencedate: 1 day after last purchase
snapshot_date = df_rfm['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = (
    df_rfm
        .groupby('CustomerID')
        .agg(
        Recency=('InvoiceDate',lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum')
    )
    .reset_index()
)
# Safety prevention. Negative/Null Monetary shouldn`t exist
rfm = rfm[rfm['Monetary'] > 0]

# Convert into quintiles
# Recency, smaller score better
recency_ranks = rfm['Recency'].rank(method='first', ascending=False)
rfm['R_Score'] = pd.qcut(recency_ranks, 5, labels=[1,2,3,4,5]).astype(int)
# Frequency, higher score better
recency_ranks = rfm['Frequency'].rank(method='first', ascending=True)
rfm['F_Score'] = pd.qcut(recency_ranks, 5, labels=[1,2,3,4,5]).astype(int)
# Monetary higher score better
recency_ranks = rfm['Monetary'].rank(method='first', ascending=True)
rfm['M_Score'] = pd.qcut(recency_ranks, 5, labels=[1,2,3,4,5]).astype(int)

# Totalscore
rfm['RFM_Score'] = (
    rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
)
rfm['RFM_Total'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

# Define Segments
def segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 4 and f >= 4:
        return 'Champions'
    if r >= 4 and f <= 2:
        return 'New Customer'
    if r <= 2 and f >= 4:
        return 'At-Risk Loyal'
    if r <= 2 and f <= 2:
        return 'Lost'
    if r >= 4 and m >= 4:
        return 'Loyal Big Spenders'
    return 'Potential Loyalist'

rfm['Segment'] = rfm.apply(segment, axis=1)

# Segment distribution
print(rfm['Segment'].value_counts().sort_values(ascending=False))

# Top Customer per Segment
top_by_segment = (
    rfm.sort_values('Monetary',ascending=False)
        .groupby('Segment')
        .head(5)
        .reset_index(drop=True)
)
print(top_by_segment.head(20))


rfm['Segment'].value_counts().plot(kind='bar', figsize=(10,8.5), title='Customers per Segment')
plt.ylabel('Count')
#plt.xlabel('Month')
#plt.subplots_adjust(top=1.02)
plt.show()

rfm.to_csv('rfm_results.csv', index=False)