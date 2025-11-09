# visualizer.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import logging
from pathlib import Path
from pandas.tseries.offsets import BDay

try:
    from config import PLOT_IMAGE_PATH
except ImportError:
    PLOT_IMAGE_PATH = "forecast_plot.png"

logger = logging.getLogger(__name__)

def plot_all_forecasts(history: pd.DataFrame, forecasts_dict: dict, best_model_name: str = None):
    try:
        history_data = history["Close"].squeeze()
        
        # Убедимся, что индекс истории — наивный DatetimeIndex
        if history_data.index.tz is not None:
            history_data.index = history_data.index.tz_localize(None)

        plt.figure(figsize=(12, 6))

        # === История: синяя сплошная ===
        plt.plot(
            history_data.index,
            history_data.values,
            label="История",
            color="blue",
            linewidth=2.5,
            linestyle="-"
        )

        # Цвета для НЕ-лучших моделей
        other_colors = {
            "Ridge": "crimson",
            "ARIMA": "hotpink",
            "LSTM": "limegreen",
            "GRU": "orange"
        }

        last_date = history_data.index[-1]  # наивный Timestamp
        forecast_start = last_date + BDay(1)

        for model_name, forecast_df in forecasts_dict.items():
            if forecast_df is None or forecast_df.empty:
                continue

            forecast_series = forecast_df["forecast"]

            # === ГАРАНТИРУЕМ: наивный DatetimeIndex ===
            if not isinstance(forecast_series.index, pd.DatetimeIndex):
                logger.warning(f"Прогноз {model_name}: индекс не DatetimeIndex. Пропускаем.")
                continue

            if forecast_series.index.tz is not None:
                forecast_series.index = forecast_series.index.tz_localize(None)

            # === ФИЛЬТРАЦИЯ: только даты >= forecast_start ===
            forecast_series = forecast_series[forecast_series.index >= forecast_start]

            if forecast_series.empty:
                logger.debug(f"Прогноз {model_name} стал пустым после фильтрации.")
                continue

            # === Стили ===
            if model_name == best_model_name:
                color = "red"
                linestyle = "-"
                linewidth = 2.5
                label = f"{model_name} (ЛУЧШАЯ)"
            else:
                color = other_colors.get(model_name, "gray")
                linestyle = "--"
                linewidth = 1.8
                label = model_name

            plt.plot(
                forecast_series.index,
                forecast_series.values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth
            )

        # Вертикальная линия — последний торговый день
        plt.axvline(
            x=last_date,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Последний торговый день"
        )

        plt.title("Прогноз цены акций")
        plt.xlabel("Дата")
        plt.ylabel("Цена, USD")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = Path(PLOT_IMAGE_PATH).resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"График сохранён: {plot_path}")
        return str(plot_path)

    except Exception as e:
        logger.exception(f"Ошибка при построении графика: {e}")
        plt.close()
        return None