import asyncio
import logging
from datetime import datetime
import numpy as np  # Import numpy here as it's used by indicator functions

from binance import AsyncClient
from aiogram import Bot


API_TOKEN = "7832605790:AAEA0qlNmOfXg4f2Fi2W8g9hSRsCw8CsKrY"
CHAT_ID = "-1002365707415"

# Список криптовалютных пар для мониторинга
PAIRS = ["SOLUSDT", "XRPUSDT", "ETHUSDT", "DOGEUSDT", "SUIUSDT", "AAVEUSDT", "WIFUSDT", "ADAUSDT", "OPUSDT", "NEARUSDT", "DOTUSDT", "TRXUSDT"]
# Словарь, сопоставляющий названия таймфреймов с их продолжительностью в минутах (Binance API использует строковые значения напрямую)
TIMEFRAMES = {"15m": 15, "1h": 60}  # 15m, 1h - это строки, которые передаются в API

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Словарь для хранения последнего отправленного сигнала для каждой пары и таймфрейма
# Это предотвращает повторную отправку дублирующихся сигналов
last_signals = {}  # Формат: (пара, название_тф, тип_сигнала) -> текст_сообщения


# --- Функции индикаторов ---

def hma(series, period=14):

    if len(series) < period:
        return np.array([])  # Not enough data

    def wma(s, length):

        if len(s) < length:
            return np.array([])
        weights = np.arange(1, length + 1)
        return np.convolve(s, weights[::-1], 'valid') / weights.sum()

    series = np.array(series, dtype=float)
    half_length = max(int(period / 2), 1)
    sqrt_length = max(int(period ** 0.5), 1)

    wma1 = wma(series, half_length)
    if wma1.size == 0: return np.array([])
    wma2 = wma(series, period)
    if wma2.size == 0: return np.array([])

    # Ensure diff calculation uses compatible lengths
    min_len = min(len(wma1), len(wma2))
    diff = 2 * wma1[-min_len:] - wma2[-min_len:]

    hma_values = wma(diff, sqrt_length)
    return hma_values


def atr(candles, period=14):

    if len(candles) < period + 1:
        return np.array([])  # Need at least period + 1 candles for first TR

    highs = np.array([c['high'] for c in candles], dtype=float)
    lows = np.array([c['low'] for c in candles], dtype=float)
    closes = np.array([c['close'] for c in candles], dtype=float)

    trs = []
    for i in range(1, len(candles)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)

    trs = np.array(trs)
    if len(trs) < period:
        return np.array([])

    atr_values = np.convolve(trs, np.ones(period) / period, mode='valid')
    return atr_values


def calculate_rsi(closes, period=14):

    if len(closes) < period + 1:
        return np.array([])

    diff = np.diff(closes)  # Calculate price changes
    gains = diff[diff > 0]
    losses = -diff[diff < 0]  # Losses are positive values

    avg_gain = np.zeros_like(diff)
    avg_loss = np.zeros_like(diff)

    # Initial average gain/loss
    avg_gain[period - 1] = np.mean(gains[:period]) if len(gains[:period]) > 0 else 0
    avg_loss[period - 1] = np.mean(losses[:period]) if len(losses[:period]) > 0 else 0

    # Exponentially smoothed average
    for i in range(period, len(diff)):
        gain = diff[i] if diff[i] > 0 else 0
        loss = -diff[i] if diff[i] < 0 else 0

        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss) / period

    rs = np.divide(avg_gain[period - 1:], avg_loss[period - 1:],
                   out=np.zeros_like(avg_gain[period - 1:]),
                   where=avg_loss[period - 1:] != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(closes, fast_period=12, slow_period=26, signal_period=9):

    if len(closes) < slow_period + signal_period:  # Need enough data for all EMAs
        return np.array([]), np.array([]), np.array([])  # macd_line, signal_line, histogram

    # Helper for Exponential Moving Average (EMA)
    def ema(s, period):
        if len(s) < period:
            return np.array([])
        ema_values = np.zeros_like(s, dtype=float)
        smoothing_factor = 2 / (period + 1)
        ema_values[period - 1] = np.mean(s[:period])  # Simple MA for initial value

        for i in range(period, len(s)):
            ema_values[i] = (s[i] - ema_values[i - 1]) * smoothing_factor + ema_values[i - 1]
        return ema_values[period - 1:]  # Return only the valid EMA values

    closes = np.array(closes, dtype=float)

    # Calculate Fast and Slow EMAs
    ema_fast = ema(closes, fast_period)
    ema_slow = ema(closes, slow_period)

    # Ensure EMAs have compatible lengths for MACD line
    min_len = min(len(ema_fast), len(ema_slow))
    if min_len == 0:
        return np.array([]), np.array([]), np.array([])

    macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]

    # Calculate Signal Line (EMA of MACD line)
    signal_line = ema(macd_line, signal_period)

    # Ensure signal_line has enough values
    if signal_line.size == 0:
        return macd_line, np.array([]), np.array([])

    # Calculate MACD Histogram
    min_len_hist = min(len(macd_line), len(signal_line))
    histogram = macd_line[-min_len_hist:] - signal_line[-min_len_hist:]

    return macd_line, signal_line, histogram


# --- Binance Data Fetching ---

async def fetch_klines(client, symbol: str, interval: str, limit: int = 200):
    """
    Fetches historical candlestick data (klines) from Binance.
    Increased limit to ensure enough data for all indicators.
    """
    try:
        klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
        data = []
        for k in klines:
            data.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
            })
        return data
    except Exception as e:
        logger.error(f"Ошибка при получении klines для {symbol} {interval}: {e}")
        return []


# --- Trading Logic with Enhanced Signals ---

def analyze_signals(candles, range_length=30, range_threshold=15, hma_period=70, rsi_period=14, macd_fast=12,
                    macd_slow=26, macd_signal=9):



    closes = np.array([c['close'] for c in candles], dtype=float)
    highs = np.array([c['high'] for c in candles], dtype=float)
    lows = np.array([c['low'] for c in candles], dtype=float)

    # Ensure enough data for all calculations
    min_required_candles = max(range_length + range_threshold + 10, hma_period, rsi_period + 1, macd_slow + macd_signal)
    if len(candles) < min_required_candles:
        logger.debug(f"Недостаточно данных для анализа. Необходимо: {min_required_candles}, Доступно: {len(candles)}")
        return None

    # --- Range Analysis ---
    lower_range = np.min(lows[-range_length:])
    upper_range = np.max(highs[-range_length:])

    def bars_since_change(arr_slice):
        """Helper to count bars since last change in min/max."""
        if not arr_slice.size: return 0
        last_val = arr_slice[-1]
        for i in range(len(arr_slice) - 2, -1, -1):
            if arr_slice[i] != last_val:
                return len(arr_slice) - 1 - i
        return len(arr_slice)

    upper_solid = bars_since_change(highs[-(range_length + range_threshold):]) >= range_threshold
    lower_solid = bars_since_change(lows[-(range_length + range_threshold):]) >= range_threshold
    ranging = upper_solid and lower_solid

    last_close = closes[-1]
    prev_close = closes[-2]  # Assuming at least 2 candles exist (checked by min_required_candles)

    # --- Indicator Calculations ---
    hma_values = hma(closes, hma_period)
    rsi_values = calculate_rsi(closes, rsi_period)
    macd_line, signal_line, macd_histogram = calculate_macd(closes, macd_fast, macd_slow, macd_signal)

    # Check if indicators returned enough data
    if hma_values.size < 2 or rsi_values.size < 1 or macd_line.size < 2 or signal_line.size < 2:
        logger.debug("Недостаточно данных для всех индикаторов.")
        return None

    hma_last = hma_values[-1]
    hma_prev = hma_values[-2]
    rsi_last = rsi_values[-1]
    macd_line_last = macd_line[-1]
    macd_line_prev = macd_line[-2]
    signal_line_last = signal_line[-1]
    signal_line_prev = signal_line[-2]

    # --- Signal Conditions ---
    signal_type = None
    signal_message = []
    current_price = last_close

    # BUY Signal Conditions
    is_bullish_breakout = last_close > upper_range and prev_close <= upper_range and ranging
    is_hma_up = hma_last > hma_prev
    is_rsi_buy = rsi_last < 70  # Not overbought
    is_macd_buy_crossover = macd_line_last > signal_line_last and macd_line_prev <= signal_line_prev

    if is_bullish_breakout and is_hma_up and is_rsi_buy and is_macd_buy_crossover:
        signal_type = "BUY"
        signal_message.append("Сигнал на покупку: Подтвержденный пробой верхнего уровня диапазона.")
        signal_message.append("  - HMA: Восходящий тренд.")
        signal_message.append(f"  - RSI ({rsi_period}): {rsi_last:.2f} (не перекуплен).")
        signal_message.append("  - MACD: MACD линия пересекла сигнальную линию вверх.")

    # SELL Signal Conditions
    is_bearish_breakout = last_close < lower_range and prev_close >= lower_range and ranging
    is_hma_down = hma_last < hma_prev
    is_rsi_sell = rsi_last > 30  # Not oversold
    is_macd_sell_crossover = macd_line_last < signal_line_last and macd_line_prev >= signal_line_prev

    if is_bearish_breakout and is_hma_down and is_rsi_sell and is_macd_sell_crossover:
        signal_type = "SELL"
        signal_message.append("Сигнал на продажу: Подтвержденный пробой нижнего уровня диапазона.")
        signal_message.append("  - HMA: Нисходящий тренд.")
        signal_message.append(f"  - RSI ({rsi_period}): {rsi_last:.2f} (не перепродан).")
        signal_message.append("  - MACD: MACD линия пересекла сигнальную линию вниз.")

    if signal_type:
        full_message = f"Текущая цена: {current_price:.4f}\n" + "\n".join(signal_message)
        return signal_type, full_message
    return None


async def main():

    bot = Bot(token=API_TOKEN)
    client = await AsyncClient.create()
    try:
        while True:
            for pair in PAIRS:
                for tf_name, tf_min in TIMEFRAMES.items():
                    logger.info(f"Проверка {pair} на {tf_name} таймфрейме...")
                    candles = await fetch_klines(client, pair, tf_name, limit=200)  # Increased limit
                    if not candles:
                        logger.warning(f"Не удалось получить данные для {pair} {tf_name}.")
                        continue

                    signal_data = analyze_signals(candles)

                    if signal_data:
                        signal_type, signal_description = signal_data
                        key = (pair, tf_name, signal_type)  # Key now includes signal type

                        message = (
                            f"⚡️ {signal_type} Сигнал по {pair} на {tf_name}:\n"
                            f"{signal_description}\n"
                            f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                        )

                        # Check if this exact signal (type and content) was sent before
                        if key not in last_signals or last_signals[key] != message:
                            await bot.send_message(chat_id=CHAT_ID, text=message)
                            last_signals[key] = message
                            logger.info(f"Отправлен сигнал: {message}")
                        else:
                            logger.info(f"Повторный {signal_type} сигнал для {pair} {tf_name}, не отправляем.")
                    await asyncio.sleep(1)  # Small delay between each pair/timeframe check
            await asyncio.sleep(60)  # Main loop delay, check every minute for new candles
    except Exception as e:
        logger.critical(f"Критическая ошибка в главном цикле бота: {e}", exc_info=True)
    finally:
        logger.info("Закрытие Binance клиента и Telegram бота...")
        await client.close()
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
