import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_customers = 200
n_products = 50
n_stores = 3
n_days = 120

rows = []
start = datetime(2025, 1, 1)
for day in range(n_days):
    date = start + timedelta(days=day)
    for _ in range(np.random.poisson(120)):
        transaction_id = f"T{day}_{np.random.randint(1e6)}"
        customer_id = f"C{np.random.randint(1, n_customers+1)}"
        product_id = f"P{np.random.randint(1, n_products+1)}"
        timestamp = date + timedelta(seconds=np.random.randint(0, 86400))
        quantity = int(np.clip(np.random.poisson(2), 1, 20))
        price = round(float(1 + np.random.rand() * 20), 2)
        store_id = f"S{np.random.randint(1, n_stores+1)}"
        promo_flag = int(np.random.rand() < 0.1)
        payment_type = np.random.choice(["cash", "card", "mobile"])
        rows.append([transaction_id, customer_id, product_id, timestamp.strftime('%Y-%m-%d %H:%M:%S'), quantity, price, store_id, promo_flag, payment_type])

pdf = pd.DataFrame(rows, columns=["transaction_id","customer_id","product_id","timestamp","quantity","price","store_id","promo_flag","payment_type"])

for _ in range(8):
    i = np.random.randint(len(pdf))
    pdf.loc[i,"quantity"] = pdf.loc[i,"quantity"] * 20
    pdf.loc[i,"price"] = pdf.loc[i,"price"] * 50

os.makedirs("data", exist_ok=True)
pdf.to_csv("data/supermarket_transactions.csv", index=False)
print("sample csv written to data/supermarket_transactions.csv")
