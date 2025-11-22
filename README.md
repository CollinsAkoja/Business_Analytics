# Business_Analytics
this project employs pytorch, tensor flow, ML and analytical tools for a samplew data set

Supermarket Analytics - packaged project

Included:
- PyTorch anomaly detection: FFN and AutoEncoder training scripts
- TensorFlow 30-day forecasting (LSTM)
- FastAPI service to serve the PyTorch FFN anomaly model
- Sample data generator: notebooks/generate_sample_data.py

Run quick test:
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) python notebooks/generate_sample_data.py
4) python src/pytorch_anomaly/train.py --csv data/supermarket_transactions.csv --epochs 4
5) python src/pytorch_anomaly/train_autoencoder.py --csv data/supermarket_transactions.csv --epochs 8
6) python src/tf_forecast/train.py --csv data/supermarket_transactions.csv --epochs 6
7) Start API (after training and having pytorch_anom_best.pth present): python src/fastapi_service/app.py
