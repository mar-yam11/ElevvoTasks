import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("D:/Elvoe/OnlineRetail/online_retail.csv")


df = df[df['CustomerID'].notnull()]


df = df[~df['InvoiceNo'].astype(str).str.contains('C')]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': lambda x: x.sum()
})


rfm.columns = ['Recency', 'Frequency', 'Monetary']


rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])


rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

def segment_customer(score):
    if score == '444':
        return 'Best Customers'
    elif score[0] == '4':
        return 'Loyal Customers'
    elif score[2] == '4':
        return 'Big Spenders'
    elif score[0] == '1':
        return 'At Risk'
    else:
        return 'Others'

rfm['Segment'] = rfm['RFM_Score'].apply(segment_customer)

#bouns
plt.figure(figsize=(10, 5))
sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index)
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title('Customer Segments based on RFM', fontsize=14)
plt.xlabel("Segment", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

