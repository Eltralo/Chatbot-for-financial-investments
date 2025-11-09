import pandas as pd
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    from pandas.tseries.offsets import BDay
except ImportError:
    torch = None
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch не установлен. LSTM отключён.")
    def train_lstm_model(*args, **kwargs):
        return None, None, None, None
    def forecast_lstm(*args, **kwargs):
        return None
    LSTMModel = None
    __all__ = ["train_lstm_model", "forecast_lstm"]
else:
    logger = logging.getLogger(__name__)

    from config import LSTM_WINDOW, LSTM_EPOCHS

    def make_lag_features(series: pd.Series, lags: int) -> pd.DataFrame:
        df = pd.DataFrame({"price": series})
        for i in range(1, lags + 1):
            df[f"lag_{i}"] = df["price"].shift(i)
        df["target"] = df["price"].shift(-1)
        return df.dropna()

    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out)
            return self.fc(out[:, -1, :])

    def train_lstm_model(train_series, test_series):
        logger.info(f"Обучение LSTM... window={LSTM_WINDOW}, epochs={LSTM_EPOCHS}")

        train_df = make_lag_features(train_series, lags=LSTM_WINDOW)
        test_df = make_lag_features(test_series, lags=LSTM_WINDOW)

        if len(train_df) < 10:
            logger.warning("Недостаточно данных для LSTM")
            return None, None, None, None

        X_train = train_df.drop(columns=["price", "target"]).values.astype(np.float32)
        y_train = train_df["target"].values.astype(np.float32)
        X_test = test_df.drop(columns=["price", "target"]).values.astype(np.float32)
        y_test = test_df["target"].values.astype(np.float32)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_test_scaled = scaler_X.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)

        model = LSTMModel(input_size=1, hidden_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(LSTM_EPOCHS):
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_tensor), y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy().ravel()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
        logger.info(f"LSTM RMSE: {rmse:.4f}, MAPE: {mape:.2%}")

        return (model, scaler_X, scaler_y, LSTM_WINDOW), y_pred, rmse, mape

    def forecast_lstm(model_tuple, full_series, steps=30):
        try:
            model, scaler_X, scaler_y, window = model_tuple
            last_price = full_series.iloc[-1]

            last_window_values = full_series.values[-window:].astype(np.float32)
            input_features_unscaled = last_window_values[::-1].reshape(1, window)
            scaled_features = scaler_X.transform(input_features_unscaled)
            seq = torch.tensor(scaled_features, dtype=torch.float32).view(1, window, 1)

            model.eval()
            preds_scaled = []
            with torch.no_grad():
                for _ in range(steps):
                    pred = model(seq)
                    preds_scaled.append(pred.item())
                    new_val = pred.view(1, 1, 1)
                    seq = torch.cat([seq[:, 1:, :], new_val], dim=1)

            preds_raw = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()

            # Коррекция: чтобы прогноз начинался с последней цены
            if len(preds_raw) > 0:
                adjustment = last_price - preds_raw[0]
                preds_corrected = preds_raw + adjustment
            else:
                preds_corrected = preds_raw

            last_date = pd.Timestamp(full_series.index.max())
            idx = pd.bdate_range(start=last_date + BDay(1), periods=steps)
            return pd.Series(preds_corrected, index=idx)

        except Exception as e:
            logger.exception(f"LSTM forecasting failed: {e}")
            return None