import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

# Load dataset
df = pd.read_csv('online_retail.csv')

"""
# Check Dataframe
print(df.info())

# Check for null values and duplicates
print(df.isnull().sum())
print("Duplicates: ", df.duplicated().sum())
"""

# Clean Dataframe
def clean_data(df):
    # Calculating Revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    # Convert Date into Datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Filter Nan CustomerIDs
    df_customers = df.dropna(subset=['CustomerID']).copy()
    df_customers['CustomerID'] = df_customers['CustomerID'].astype(int)

    # Calculate revenue per customer
    top_customers = df_customers.groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False)
    #(top_customers.head(10))

    # Calculate revenue per country
    revenue_country = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
    #print(revenue_country.head(10))

    # Calculate revenue per timeslot
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    revenue_month = df.groupby('InvoiceMonth')['Revenue'].sum()
    #print(revenue_month)

    # Filter out negative Quantity (Retour)
    df_positive = df[df['Quantity'] > 0]

    # Products per Invoice
    basket = df_positive.groupby('InvoiceNo')['StockCode'].apply(list)
    #print(basket.head())


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
    return df_rfm

# Analysis RFM
def analyse_data_rfm(df_rfm):
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
    #print(rfm['Segment'].value_counts().sort_values(ascending=False))

    # Top Customer per Segment
    top_by_segment = (
        rfm.sort_values('Monetary',ascending=False)
            .groupby('Segment')
            .head(5)
            .reset_index(drop=True)
    )
    #print(top_by_segment.head(20))

    return rfm

# Analysis Top Products
def top_products(df_sales, top_n=10):
    # Aggregate Products by Keypoints
    prod_agg = (
        df_sales.groupby(['StockCode', 'Description'])
                .agg(total_revenue=('Revenue', 'sum'),
                     total_qty=('Quantity', 'sum'),
                     n_invoices=('InvoiceNo', 'unique'),
                     avg_unitprice=('UnitPrice', 'mean'))
                .sort_values('total_revenue', ascending=False)
                .reset_index()
    )
    return prod_agg.head(top_n)

# Analysis Co-Purchase
def co_purchase_analysis(df_sales, min_pair_baskets=15):
    # Sort Products by InvoiceNumber
    transactions = df_sales.groupby('InvoiceNo')['StockCode'].apply(lambda s: set(s.astype(str)))
    # Initiate Counter
    n_baskets = len(transactions)
    item_counter = Counter()
    pair_counter = Counter()

    # Create Pairs from Purchases
    for items in transactions:
        item_counter.update(items)
        for a,b in combinations(sorted(items),2):
            pair_counter[frozenset((a,b))] += 1

    # Create Dataframe
    rows = []
    for pair, cnt in pair_counter.items():
        a,b = tuple(pair)
        supp_ab = cnt/n_baskets                 # Support for Pair  - Pairs compared to total baskets
        supp_a = item_counter[a]/n_baskets      # Support for Item A- Item A compared to total baskets
        supp_b = item_counter[b]/n_baskets      # Support for Item B- Item B compared to total baskets
        conf_a2b = cnt/item_counter[a]          # Confidence for Item B if A was bought - Pairs compared to baskets with Item A
        conf_b2a = cnt/item_counter[b]          # Confidence for Item A if B was bought - Pairs compared to baskets with Item B
        lift = supp_ab/(supp_a*supp_b) if supp_a>0 and supp_b>0 else float('nan')      #- Calculating coincidence( 1=coincidence, >1 positive association, <1 negative association)
        rows.append({'item_a':a, 'item_b':b, 'pair_count':cnt, 'support_ab':supp_ab,
                     'confidence_a_to_b':conf_a2b, 'confidence_b_2_a': conf_b2a, 'lift':lift})

    # Create Dataframe for Pairs
    pairs_df = pd.DataFrame(rows)
    #pairs_df = pairs_df[pairs_df['pair_count']>=min_pair_baskets]
    return pairs_df

# RFM Plot
def plot_rfm(rfm):
    rfm['Segment'].value_counts().plot(kind='bar', figsize=(10,8.5), title='Customers per Segment')
    plt.ylabel('Count')
    #plt.xlabel('Month')
    #plt.subplots_adjust(top=1.02)
    plt.show()

# Top Products Plot
def plot_top_products(prod_agg):
    prod_agg.plot(kind='bar', x='Description', y='total_revenue', figsize=(8,5),
                  title='Top Products by Revenue')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

# ProductPairs Plot
def plot_pairs(pairs_df):
    top_pairs = pairs_df.sort_values('lift', ascending=False).head(10)
    # Create Labels A & B
    top_pairs['pair'] = top_pairs['item_a'] + ' & ' + top_pairs['item_b']

    # Plot
    plt.figure(figsize=(10,6))
    plt.barh(top_pairs['pair'], top_pairs['lift'], color='skyblue')
    plt.xlabel('Lift')
    plt.title('Top 10 Product Pairs by Lift')
    plt.gca().invert_yaxis()    # Highest Value at the Top
    plt.tight_layout()
    plt.show()

# Main
# RFM-Analysis
df_rfm = clean_data(df)
rfm = analyse_data_rfm(df_rfm)
rfm.to_csv('rfm_results.csv', index=False)

# Analysis Top Products
df_sales = df_rfm.copy()
prod_agg = top_products(df_sales)
prod_agg.to_csv('prod_agg_results.csv', index=False)

# Analysis ProductPairs
pairs_df = co_purchase_analysis(df_sales)
pairs_df.head(10)

# Plots
plot_rfm(rfm)
plot_top_products(prod_agg)
plot_pairs(pairs_df)