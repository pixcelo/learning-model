import ccxt
import pandas as pd

# 取引所の設定
exchange = ccxt.bybit({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

# 価格データの取得
symbol = 'BTC/USDT'
timeframe = '15m'
since = exchange.parse8601('2020-01-01T00:00:00Z')
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)

# データフレームに変換し、CSVファイルに保存
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv('data.csv', index=False)