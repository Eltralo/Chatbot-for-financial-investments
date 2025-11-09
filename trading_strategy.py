import pandas as pd
import logging

logger = logging.getLogger(__name__)

def find_local_extrema(series: pd.Series):
    """
    Находит локальные минимумы и максимумы.
    Минимум: цена ниже, чем за 1 и 2 дня до/после.
    Максимум: цена выше, чем за 1 и 2 дня до/после.
    """
    if len(series) < 5:
        return [], []

    prices = series.values
    minima = []
    maxima = []

    for i in range(2, len(prices) - 2):
        if (prices[i] < prices[i-2] and prices[i] < prices[i-1] and
            prices[i] < prices[i+1] and prices[i] < prices[i+2]):
            minima.append(i)
        elif (prices[i] > prices[i-2] and prices[i] > prices[i-1] and
              prices[i] > prices[i+1] and prices[i] > prices[i+2]):
            maxima.append(i)
    
    return minima, maxima

def simulate_extremum_strategy(forecast_series: pd.Series, initial_capital: float):
    """
    Стратегия на основе локальных экстремумов:
    - Покупаем ВСЁ на локальных минимумах
    - Продаём ВСЁ на локальных максимумах
    """
    if initial_capital <= 0:
        return 0.0, "Сумма инвестиций должна быть больше нуля."

    prices = forecast_series.dropna()
    if len(prices) < 5:
        return 0.0, "Недостаточно данных для поиска экстремумов (требуется минимум 5 дней)."

    min_indices, max_indices = find_local_extrema(prices)
    if not min_indices and not max_indices:
        return 0.0, "Локальные экстремумы не обнаружены. Стратегия не применима."

    actions = []
    cash = float(initial_capital)
    shares = 0.0
    last_action = None  # 'buy' или 'sell'

    # Собираем все точки событий
    events = []
    for i in min_indices:
        events.append((i, 'min'))
    for i in max_indices:
        events.append((i, 'max'))
    events.sort()

    for i, typ in events:
        date_str = prices.index[i].strftime('%Y-%m-%d')
        price = prices.iloc[i]

        if typ == 'min' and (last_action is None or last_action == 'sell'):
            if cash > 0:
                new_shares = cash / price
                shares += new_shares
                cash = 0.0
                actions.append(f"Покупка на локальном минимуме: {new_shares:.4f} акций по ${price:.2f} ({date_str})")
                last_action = 'buy'
        elif typ == 'max' and last_action == 'buy':
            if shares > 0:
                revenue = shares * price
                cash += revenue
                actions.append(f"Продажа на локальном максимуме: {shares:.4f} акций по ${price:.2f} ({date_str})")
                shares = 0.0
                last_action = 'sell'

    # Финальная продажа, если остались акции
    if shares > 0:
        final_date = prices.index[-1].strftime('%Y-%m-%d')
        final_price = prices.iloc[-1]
        revenue = shares * final_price
        cash += revenue
        actions.append(f"Финальная продажа: {shares:.4f} акций по ${final_price:.2f} ({final_date})")

    profit = cash - initial_capital
    report = "\n".join(actions) if actions else "Стратегия не сгенерировала действий."
    return profit, report