import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import optuna
import logging
from pandas.tseries.offsets import BDay

logger = logging.getLogger(__name__)

def create_features(price_series: pd.Series, lags: int = 5) -> pd.DataFrame:
    """
    Создаёт признаки **без утечки будущего**.
    Только лаги: lag_1 = цена вчера, lag_2 = позавчера и т.д.
    """
    df = pd.DataFrame({'price': price_series})
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['price'].shift(i)
    df['target'] = df['price'].shift(-1)  # цена завтра
    return df.dropna()

def train_ridge_regression(train_prices, test_prices, lags=5):
    logger.info("Обучение Ridge Regression...")
    train_df = create_features(train_prices, lags)
    test_df = create_features(test_prices, lags)
    
    if len(train_df) < 20 or len(test_df) == 0:
        raise ValueError("Недостаточно данных для Ridge Regression")
    
    # Признаки = только лаги
    feature_cols = [f for f in train_df.columns if f.startswith('lag_')]
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.001, 100.0, log=True)
        model = Pipeline([
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_alpha = study.best_params['alpha']
    
    model = Pipeline([
        ('ridge', Ridge(alpha=best_alpha, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Ridge Regression RMSE: {rmse:.4f}, MAPE: {mape:.2%}")
    return model, y_pred, rmse, mape

def generate_ridge_regression_forecast(model, full_price_series, forecast_horizon=30, lags=5):
    """
    Рекурсивный прогноз на основе лагов.
    """
    logger.info("Generating Ridge Regression forecast...")
    
    # Начинаем с последних lags цен
    last_prices = full_price_series.iloc[-lags:].tolist()
    forecast = []
    
    for _ in range(forecast_horizon):
        X = np.array([last_prices]).astype(float)
        next_price = model.predict(X)[0]
        forecast.append(next_price)
        
        # Обновляем лаги: сдвигаем и добавляем новую цену
        last_prices = [next_price] + last_prices[:-1]
    
    # Генерируем индексы рабочих дней
    last_date = pd.Timestamp(full_price_series.index.max())
    idx = pd.bdate_range(start=last_date + BDay(1), periods=forecast_horizon)
    return pd.Series(forecast, index=idx)