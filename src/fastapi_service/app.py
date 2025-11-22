from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch, numpy as np
import uvicorn
import pandas as pd
from pathlib import Path

app = FastAPI(title="Supermarket Anomaly Service")

class RowInput(BaseModel):
    quantity: float
    price: float
    store_id: str
    payment_type: str
    promo_flag: int

MODEL_PATH = Path("pytorch_anom_best.pth")
SCALER_PATH = Path("scaler.npy")

model = None
scaler = None
store_cols = None
pay_cols = None

def load_model():
    global model, scaler, store_cols, pay_cols
    if not MODEL_PATH.exists():
        print("Model not found, serve limited endpoints. Train model first.")
        return
    from src.pytorch_anomaly.model import FFN
    # load a tiny dataset to get dims and column order
    from src.pytorch_anomaly.dataset import AnomalyDataset
    ds = AnomalyDataset("data/supermarket_transactions.csv")
    input_dim = ds.X.shape[1]
    model_local = FFN(input_dim)
    ck = torch.load(MODEL_PATH, map_location="cpu")
    model_local.load_state_dict(ck["model_state_dict"])
    model_local.eval()
    model = model_local
    # save some helpers
    scaler = ds.scaler
    return model, scaler

@app.on_event("startup")
def startup_event():
    global model, scaler
    model, scaler = load_model()

@app.post("/predict_anomaly")
def predict(row: RowInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    # build features in same order as dataset: numeric (quantity,price), store ohe, pay ohe, promo
    # to keep the service simple, read dataset to get ohe columns
    df = pd.read_csv("data/supermarket_transactions.csv", parse_dates=["timestamp"])
    store_ohe = pd.get_dummies(df["store_id"], prefix="store")
    pay_ohe = pd.get_dummies(df["payment_type"], prefix="pay")
    # build template row
    num = np.array([[row.quantity, row.price]])
    num_s = scaler.transform(num)
    store_cols = store_ohe.columns.tolist()
    pay_cols = pay_ohe.columns.tolist()
    store_vec = np.zeros((1, len(store_cols)))
    pay_vec = np.zeros((1, len(pay_cols)))
    # try to set correct index if present
    s_col = f"store_{row.store_id}"
    p_col = f"pay_{row.payment_type}"
    if s_col in store_cols:
        store_vec[0, store_cols.index(s_col)] = 1
    if p_col in pay_cols:
        pay_vec[0, pay_cols.index(p_col)] = 1
    promo = np.array([[row.promo_flag]])
    X = np.hstack([num_s, store_vec, pay_vec, promo]).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        prob = torch.sigmoid(logits).item()
    return {"anomaly_score": float(prob)}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
