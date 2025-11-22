import argparse
from src.tf_forecast.dataset import make_tf_dataset
from src.tf_forecast.model import build_lstm_forecast

def train(args):
    ds = make_tf_dataset(args.csv, batch_size=args.batch_size)
    model = build_lstm_forecast(input_len=30)
    callbacks = [
        __import__('tensorflow').keras.callbacks.ModelCheckpoint(args.checkpoint, save_best_only=True),
        __import__('tensorflow').keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    model.fit(ds, epochs=args.epochs, callbacks=callbacks)
    model.save(args.saved_model)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/supermarket_transactions.csv')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default='tf_forecast_ckpt.h5')
    parser.add_argument('--saved_model', type=str, default='tf_forecast_saved')
    args = parser.parse_args()
    train(args)
    print('Training completed.')