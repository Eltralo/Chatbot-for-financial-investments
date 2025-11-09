import os
from pathlib import Path

# --- КОНФИГУРАЦИЯ БОТА ---
# Замените на ваш токен из @BotFather
BOT_TOKEN = ""

# --- ПУТИ ---
BASE_DIR = Path(__file__).resolve().parent
PLOT_IMAGE_PATH = BASE_DIR / "forecast_plot.png"
LOG_FILE = BASE_DIR / "logs.csv"

# --- КОНФИГУРАЦИЯ МОДЕЛЕЙ ---
TEST_SET_DAYS = 30
FORECAST_DAYS = 30
LSTM_WINDOW = 20
LSTM_EPOCHS = 40  # увеличено для лучшего обучения

# --- КОНФИГУРАЦИЯ ДАННЫХ ---
DATA_PERIOD = "2y"