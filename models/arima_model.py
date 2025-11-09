# arima_model.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import itertools

logger = logging.getLogger(__name__)

def train_arima_model(train_series, test_series):
    """
    Обучает ARIMA-модель с автоматическим подбором p и q, при фиксированном d=1.
    Если все модели дают плохой прогноз — используем fallback (5,1,2).
    """
    train_data = train_series.values
    test_data = test_series.values

    if len(train_data) < 30:
        raise ValueError("Недостаточно данных для ARIMA")

    d = 1

    # Диапазоны для p и q
    p_vals = range(1, 6)   # 1–5
    q_vals = range(0, 4)   # 0–3

    best_aic = float('inf')
    best_order = None
    best_model = None

    logger.info(f"Подбор лучшего порядка ARIMA (d={d})...")

    for p in p_vals:
        for q in q_vals:
            try:
                model = ARIMA(train_data, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                fitted = model.fit()
                aic = fitted.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = fitted
            except Exception:
                continue

    # Если не нашли хорошую модель — используем fallback
    if best_model is None:
        logger.warning("Не удалось подобрать ARIMA-модель. Используем fallback (5,1,2).")
        model = ARIMA(train_data, order=(5, 1, 2))
        best_model = model.fit()
        best_order = (5, 1, 2)

    logger.info(f"Лучший порядок ARIMA: p={best_order[0]}, d={best_order[1]}, q={best_order[2]} (AIC={best_aic:.2f})")

    # Прогноз на тестовые данные
    try:
        end_idx = len(train_data) + len(test_data) - 1
        preds = best_model.predict(start=len(train_data), end=end_idx)
        if len(preds) != len(test_data):
            raise ValueError("Несоответствие длины прогноза")
    except Exception:
        # Фолбэк: пошаговое прогнозирование
        preds = []
        temp_model = best_model
        for i in range(len(test_data)):
            pred = temp_model.forecast(steps=1)[-1]
            preds.append(pred)
            # Для пошагового прогноза можно обновлять модель, но это медленно
        preds = np.array(preds)

    rmse = np.sqrt(mean_squared_error(test_data, preds))
    mape = mean_absolute_percentage_error(test_data, preds)
    logger.info(f"ARIMA RMSE: {rmse:.4f}, MAPE: {mape:.2%}")

    return best_model, preds, rmse, mape

def forecast_arima(model, steps=30):
    """Генерирует прогноз вне выборки."""
    try:
        forecast = model.forecast(steps=steps)
        return forecast.tolist()
    except Exception as e:
        logger.error(f"Ошибка прогнозирования ARIMA: {e}")
        return None