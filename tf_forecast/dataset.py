import pandas as pd
import numpy as np
import tensorflow as tf

def build_daily_agg(csv_path, min_days=60):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df["date"] = df["timestamp"].dt.date
    df["amount"] = df["quantity"] * df["price"]
    daily = df.groupby(["product_id","date"])["quantity"].sum().reset_index()
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max())
    products = []
    for pid, g in daily.groupby("product_id"):
        s = g.set_index("date").reindex(all_dates, fill_value=0)["quantity"]
        if len(s) >= min_days:
            products.append((pid, s.values))
    return products, all_dates

def windows_for_product(series, input_len=30, horizon=1):
    X = []
    Y = []
    L = len(series)
    for i in range(0, L - input_len - horizon + 1):
        X.append(series[i:i+input_len])
        Y.append(series[i+input_len:i+input_len+horizon])
    return np.array(X), np.array(Y)

def make_tf_dataset(csv_path, batch_size=64):
    products, dates = build_daily_agg(csv_path)
    Xs = []
    Ys = []
    for pid, series in products:
        x,y = windows_for_product(series, input_len=30, horizon=1)
        if len(x):
            Xs.append(x)
            Ys.append(y)
    if not Xs:
        raise ValueError('no series found')
    X = np.vstack(Xs)
    Y = np.vstack(Ys).squeeze(-1)
    ds = tf.data.Dataset.from_tensor_slices((X[..., np.newaxis].astype(np.float32), Y.astype(np.float32)))
    ds = ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

