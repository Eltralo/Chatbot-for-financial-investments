import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

try:
    from config import DATA_PERIOD 
except ImportError:
    DATA_PERIOD = "2y"

logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str, period: str = DATA_PERIOD) -> pd.DataFrame:
    """
    Загружает и очищает исторические данные акций с Yahoo Finance.
    Возвращает DataFrame с уникальным дат-тайм индексом и без пропусков.
    """
    try:
        ticker = ticker.upper().strip()
        
        end = datetime.now()
        start = end - timedelta(days=730)
        
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            timeout=10
        )
        
        if df.empty:
            raise ValueError(f"Данные для тикера {ticker} не найдены")
        
        # Очистка индекса
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Выбор только Close и удаление строк, где Close отсутствует
        if "Close" not in df.columns:
            raise ValueError(f"В данных для {ticker} отсутствует колонка 'Close'")
            
        result = df[["Close"]].copy()
        result = result.dropna()
        
        # Удаление дубликатов по индексу (оставляем последнюю запись)
        result = result[~result.index.duplicated(keep='last')]
        
        # Сортировка по дате (на случай неправильного порядка)
        result = result.sort_index()
        
        # Минимум 90 записей после очистки
        if len(result) < 90:
            raise ValueError(f"Недостаточно данных для {ticker} после очистки. "
                           f"Осталось {len(result)} записей (минимум 90).")
            
        return result
        
    except Exception as e:
        logger.error(f"Не удалось загрузить и очистить данные для {ticker}: {e}")
        raise ValueError(f"Не удалось загрузить данные для {ticker}. "
                        f"Проверьте корректность тикера или повторите попытку позже.")