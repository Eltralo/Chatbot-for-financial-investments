import pandas as pd
from pathlib import Path
import logging

try:
    from config import LOG_FILE 
except ImportError:
    LOG_FILE = Path(__file__).resolve().parent / "logs.csv"
    
logger = logging.getLogger(__name__)

class LogManager:
    """Осуществляет логирование запросов пользователей и результатов прогнозов в CSV-файл (Этап 5)."""
    
    def __init__(self, log_path: Path = LOG_FILE):
        self.log_path = log_path
        self.columns = [
            'user_id', 'datetime', 'ticker', 'investment_amount', 
            'best_model', 'metric_value', 'estimated_profit_usd'
        ]
        if not self.log_path.exists():
            self._create_log_file()

    def _create_log_file(self):
        """Создаёт CSV-файл логов с заголовками, если он ещё не существует."""
        try:
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.log_path, index=False)
        except Exception as e:
            logger.error(f"Не удалось создать файл логов: {e}")

    def log_request(self, user_id, ticker, amount, best_model, metric_value, profit):
        """Логирует результаты одного запроса."""
        new_entry = {
            'user_id': user_id,
            'datetime': pd.Timestamp.now().isoformat(),
            'ticker': ticker.upper(),
            'investment_amount': amount,
            'best_model': best_model,
            'metric_value': f"{metric_value:.4f}",
            'estimated_profit_usd': f"{profit:.2f}"
        }
        
        try:
            df = pd.read_csv(self.log_path)
            new_row_df = pd.DataFrame([new_entry], columns=self.columns)
            df = pd.concat([df, new_row_df], ignore_index=True)
            df.to_csv(self.log_path, index=False)
        except Exception as e:
            logger.error(f"Не удалось записать лог для {ticker}: {e}")

# Инициализация LogManager
log_manager = LogManager()