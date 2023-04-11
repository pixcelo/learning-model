import ccxt
import pandas as pd
import datetime
import os
import logging

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# 取引所の設定
exchange = ccxt.bybit({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

def fetch_ohlcv_with_timeframe(exchange, symbol, timeframe, since, until):
    all_candles = []
    while since < until:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            break
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            break

        if len(candles) == 0:
            break
        since = candles[-1][0] + exchange.parse_timeframe(timeframe) * 1000
        all_candles += candles
    return all_candles

# 価格データの取得
symbol = 'BTC/USDT'
timeframe = '15m'
start_date = '2021-08-01T00:00:00Z'
end_date = '2021-12-31T00:00:00Z'
since = exchange.parse8601(start_date)
until = exchange.parse8601(end_date)

ohlcv = fetch_ohlcv_with_timeframe(exchange, symbol, timeframe, since, until)

# データが空でないことを確認
if len(ohlcv) == 0:
    logger.warning("No data fetched.")
    exit()

# データフレームに変換
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['symbol'] = symbol
df['timeframe'] = timeframe

# ファイル名の作成
symbol_name = symbol.replace("/", "")
start_date_str = start_date[:10].replace("-", "")
end_date_str = end_date[:10].replace("-", "")
csv_file_name = f"{symbol_name}_{timeframe}_{start_date_str}_{end_date_str}.csv"

# 保存先フォルダを作成（存在しない場合）
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# CSVファイルに保存
csv_file_path = os.path.join(data_dir, csv_file_name)
df.to_csv(csv_file_path, index=False)
