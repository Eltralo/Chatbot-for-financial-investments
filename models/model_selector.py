# models/model_selector.py

import pandas as pd
import logging
from pandas.tseries.offsets import BDay

try:
    from config import TEST_SET_DAYS, FORECAST_DAYS
except ImportError:
    TEST_SET_DAYS = 30
    FORECAST_DAYS = 30

logger = logging.getLogger(__name__)

def evaluate_all_models(df: pd.DataFrame):
    split_idx = len(df) - TEST_SET_DAYS
    if split_idx < 60:
        raise ValueError("Insufficient data. Need at least 90 trading days for a 30-day split and some history.")

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    # Убедимся, что индекс — DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Индекс данных не DatetimeIndex. Преобразуем...")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Не удалось преобразовать индекс в DatetimeIndex: {e}")

    train_s = train["Close"].squeeze()
    test_s = test["Close"].squeeze()
    full_s = df["Close"].squeeze()
    last_date = full_s.index[-1]

    results = []
    forecasts = {}

    # Ridge
    try:
        from .ml_model import train_ridge_regression, generate_ridge_regression_forecast
        model_tuple, preds, rmse, mape = train_ridge_regression(train_s, test_s, lags=5)
        results.append(("Ridge", model_tuple, preds, rmse, mape))
        fc = generate_ridge_regression_forecast(model_tuple, full_s, forecast_horizon=FORECAST_DAYS, lags=5)
        if fc is not None:
            fc = fc[fc.index > last_date]
            if not fc.empty:
                forecasts["Ridge"] = pd.DataFrame({"forecast": fc})
    except Exception as e:
        logger.error(f"Ridge failed: {e}", exc_info=True)

    # ARIMA
    try:
        from .arima_model import train_arima_model, forecast_arima
        model_tuple, preds, rmse, mape = train_arima_model(train_s, test_s)
        results.append(("ARIMA", model_tuple, preds, rmse, mape))
        fc_list = forecast_arima(model_tuple, steps=FORECAST_DAYS)
        if fc_list is not None:
            idx = pd.bdate_range(start=last_date + BDay(1), periods=FORECAST_DAYS)
            fc_series = pd.Series(fc_list, index=idx)
            fc_series = fc_series[fc_series.index > last_date]
            if not fc_series.empty:
                forecasts["ARIMA"] = pd.DataFrame({"forecast": fc_series})
    except Exception as e:
        logger.error(f"ARIMA failed: {e}", exc_info=True)

    # LSTM
    try:
        from .nn_model import train_lstm_model, forecast_lstm
        model_tuple, preds, rmse, mape = train_lstm_model(train_s, test_s)
        if model_tuple is not None:
            results.append(("LSTM", model_tuple, preds, rmse, mape))
            fc = forecast_lstm(model_tuple, full_s, steps=FORECAST_DAYS)
            if fc is not None:
                fc = fc[fc.index > last_date]
                if not fc.empty:
                    forecasts["LSTM"] = pd.DataFrame({"forecast": fc})
    except Exception as e:
        logger.error(f"LSTM failed: {e}", exc_info=True)

    if not results:
        raise RuntimeError("All models failed to train or evaluate successfully.")

    best_name, _, _, best_rmse, _ = min(results, key=lambda x: x[3])
    if best_name not in forecasts or forecasts[best_name].empty:
        raise RuntimeError(f"Forecast for best model {best_name} not generated successfully.")

    best_forecast_df = forecasts[best_name]
    all_results = {name: {'rmse': r, 'mape': m} for name, _, _, r, m in results}
    return best_name, best_forecast_df, best_rmse, all_results, forecasts