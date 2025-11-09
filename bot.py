import logging
import os
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, FSInputFile
from aiogram.client.default import DefaultBotProperties
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from config import BOT_TOKEN
    from log_manager import log_manager
    from data_loader import fetch_stock_data
    from models.model_selector import evaluate_all_models
    from trading_strategy import simulate_extremum_strategy
    from visualizer import plot_all_forecasts
except ImportError as e:
    logger.error(f"Ошибка импорта компонентов: {e}")
    exit(1)

if not BOT_TOKEN or "YOUR_BOT_TOKEN" in BOT_TOKEN:
    logger.error("BOT_TOKEN не настроен в config.py.")
    exit(1)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher(storage=MemoryStorage())

class StockPrediction(StatesGroup):
    waiting_for_ticker = State()
    waiting_for_amount = State()

@dp.message(lambda msg: msg.text in ["/start", "/help"])
async def send_welcome(message: Message):
    text = (
        "Привет!\n\n"
        "Я учебный чат-бот для анализа и прогнозирования акций на основе временных рядов.\n"
        "Отправьте команду /predict, чтобы начать.\n\n"
        "Примеры популярных тикеров:\n"
        "• AAPL — Apple\n"
        "• MSFT — Microsoft\n"
        "• GOOGL — Alphabet (Google)\n"
        "• AMZN — Amazon\n"
        "• TSLA — Tesla\n"
        "• NVDA — NVIDIA\n"
        "• JPM — JPMorgan Chase\n"
        "• BAC — Bank of America\n"
        "• NFLX — Netflix\n"
        "• DIS — Disney\n\n"
        "Процесс включает: загрузку данных, обучение моделей, выбор лучшей, прогноз на 30 дней и торговую стратегию."
    )
    await message.answer(text)

@dp.message(lambda msg: msg.text == "/predict")
async def start_prediction(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(StockPrediction.waiting_for_ticker)
    await message.answer("Введите тикер акции (например, AAPL):")

@dp.message(StockPrediction.waiting_for_ticker)
async def process_ticker(message: Message, state: FSMContext):
    ticker = message.text.strip().upper()
    if not ticker or not ticker.isalnum():
        await message.answer("Ошибка: неверный формат тикера. Используйте только буквы и цифры (например, AAPL).")
        return

    try:
        df = fetch_stock_data(ticker)
        if df.empty:
            raise ValueError("Тикер не найден")
    except Exception as e:
        await message.answer(f"Ошибка: тикер '{ticker}' не найден или недоступен. Попробуйте другой (например, AAPL, MSFT, GOOGL).")
        return

    await state.update_data(ticker=ticker)
    await state.set_state(StockPrediction.waiting_for_amount)
    await message.answer(f"Тикер: {ticker}.\nТеперь введите сумму условных инвестиций в USD (например, 10000):")

@dp.message(StockPrediction.waiting_for_amount)
async def process_amount(message: Message, state: FSMContext):
    data = await state.get_data()
    ticker = data.get('ticker', 'UNKNOWN')
    
    try:
        investment_amount = float(message.text.strip().replace(',', ''))
        if investment_amount <= 0:
            await message.answer("Ошибка: сумма должна быть больше нуля.")
            return
        if investment_amount > 1000000:
            await message.answer("Ошибка: сумма слишком большая. Максимум — 1 000 000 USD.")
            return
        investment_amount = round(investment_amount, 2)
    except ValueError:
        await message.answer("Ошибка: введите корректное число (например, 10000).")
        return

    status_msg = await message.answer(
        f"Запускаю анализ для {ticker} с суммой {investment_amount:,.2f} USD...\n\n"
        "Обучаю модели...\n"
        "Генерирую прогнозы...\n"
        "Строю график...\n"
        "Рассчитываю прибыль по стратегии...\n\n"
        "Это может занять до минуты. Пожалуйста, подождите."
    )

    try:
        df = fetch_stock_data(ticker)
        current_close_price = df["Close"].iloc[-1].item()

        best_name, best_forecast_df, best_rmse, all_results, all_forecasts = evaluate_all_models(df)

        if best_forecast_df is None or best_forecast_df.empty:
            raise RuntimeError("Не удалось сгенерировать прогноз.")

        plot_path = plot_all_forecasts(df, all_forecasts, best_name)
        forecast_series = best_forecast_df["forecast"]
        profit, trade_report = simulate_extremum_strategy(forecast_series, investment_amount)

        report = f"АНАЛИЗ ЗАВЕРШЕН: {ticker}\n\n"
        report += f"Лучшая модель: {best_name}\n\n"
        report += "Метрики всех моделей:\n"
        for name in ["Ridge", "ARIMA", "LSTM"]:
            if name in all_results:
                r = all_results[name]['rmse']
                m = all_results[name]['mape'] * 100
                mark = " (ЛУЧШАЯ)" if name == best_name else ""
                report += f"- {name}{mark}: RMSE = {r:.4f}, MAPE = {m:.2f}%\n"
            else:
                report += f"- {name}: не обучена\n"
        report += "\n"
        if not forecast_series.empty:
            final_price = float(forecast_series.iloc[-1])
            change_pct = (final_price - current_close_price) / current_close_price * 100
            direction = "рост" if change_pct >= 0 else "падение"
            report += f"Прогнозируется {direction} на {change_pct:.2f}% "
            report += f"(с {current_close_price:.2f} до {final_price:.2f}).\n\n"
        else:
            report += "Прогноз недоступен.\n\n"
        report += f"Торговая стратегия (инвестиции: {investment_amount:,.2f}):\n"
        report += trade_report
        report += f"\n\nИтоговая прибыль: {profit:,.2f} ({profit / investment_amount * 100:+.2f}%)\n\n"
        report += "⚠️ ВАЖНО: Этот бот является учебным проектом. Прогнозы не следует использовать для принятия реальных инвестиционных решений."

        log_manager.log_request(
            user_id=message.from_user.id,
            ticker=ticker,
            amount=investment_amount,
            best_model=best_name,
            metric_value=best_rmse,
            profit=profit
        )

        await bot.edit_message_text(
            report,
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        if plot_path and os.path.exists(plot_path):
            await bot.send_photo(
                message.chat.id,
                FSInputFile(plot_path),
                caption="График прогноза"
            )

        await state.clear()

    except (ValueError, RuntimeError) as e:
        await bot.edit_message_text(
            f"Анализ для {ticker} завершился с ошибкой: {e}\nПопробуйте /predict снова.",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )
        await state.clear()
    except Exception as e:
        logger.exception("Неожиданная ошибка бота")
        await bot.edit_message_text(
            "Произошла системная ошибка. Попробуйте /predict снова.",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )
        await state.clear()

@dp.message()
async def handle_unknown_message(message: Message):
    await message.answer(
        "Извините, я не понимаю эту команду.\n"
        "Используйте:\n"
        "/start - начало работы\n"
        "/predict - анализ акций\n"
        "/help - справка"
    )

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass