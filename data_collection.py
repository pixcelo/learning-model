import os
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import configparser

# 設定ファイルの読み込み
config = configparser.ConfigParser()
config.read('config.ini')

# APIキーとシークレットキーの取得
api_key = config.get('bybit', 'api_key')
secret_key = config.get('bybit', 'secret_key')

exchange = ccxt.bybit({
    'apiKey': api_key,
    'secret': secret_key,
})

# カラム名にprefixを付与する関数
def add_prefix_to_columns(df, prefix):
    df.columns = [f'{prefix}_{column}' for column in df.columns]
    return df

# 複数の時間足を取得するための関数
def fetch_ohlcv_multiple_timeframes(exchange, symbol, timeframes, start_date, end_date):
    data = {}
    for timeframe in timeframes:
        print(f"Fetching {symbol} {timeframe} data...")
        data[timeframe] = []
        current_date = start_date
        while current_date < end_date:
            fetched_data = exchange.fetch_ohlcv(symbol, timeframe, since=current_date)
            if len(fetched_data) == 0:
                break
            data[timeframe].extend(fetched_data)
            current_date = fetched_data[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            time.sleep(exchange.rateLimit / 1000 * 2) # APIのレート制限に対応するための遅延処理
        print(f"{symbol} {timeframe} data fetched.")
    return data

# 精算情報を取得するための関数
def fetch_liquidations(exchange, symbol, start_date, end_date):
    print(f"Fetching {symbol} liquidations data...")
    # 以下のAPIエンドポイントを利用して精算情報を取得します。
    # ご利用の取引所によっては、別のエンドポイントを利用する必要がある場合があります。
    # 公式ドキュメントを確認してください。
    url = f'https://api.bybit.com/v2/public/liq-records?symbol={symbol}&from={start_date}&to={end_date}'
    liquidations = pd.read_json(url)
    print(f"{symbol} liquidations data fetched.")
    return liquidations

# 日時をUNIXタイムスタンプに変換するための関数
def to_unix_timestamp(date):
    return int(date.timestamp() * 1000)

# データをCSVファイルに保存するための関数
def save_data_to_csv(data, file_name):
    df = pd.DataFrame(data)
    column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df.columns = column_names
    file_path = os.path.join("data", file_name)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def collect_historical_data():
    # 取得するデータの設定
    symbol = 'BTC/USDT'
    timeframes = ['15m', '1h', '4h']
    start_date = datetime(2021, 8, 1)
    end_date = datetime(2021, 12, 31)
    unix_start_date = to_unix_timestamp(start_date)
    unix_end_date = to_unix_timestamp(end_date)

    # 複数の時間足のデータを取得
    ohlcv_data = fetch_ohlcv_multiple_timeframes(exchange, symbol, timeframes, unix_start_date, unix_end_date)

    # データフレームに変換し、カラム名にprefixを付与
    dfs = {}
    for timeframe in timeframes:
        dfs[timeframe] = pd.DataFrame(ohlcv_data[timeframe])
        dfs[timeframe].columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        dfs[timeframe] = add_prefix_to_columns(dfs[timeframe], timeframe)

    # データをCSVファイルに保存
    for timeframe in timeframes:
        file_name = f'BTCUSDT_{timeframe}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
        dfs[timeframe].to_csv(os.path.join("data", file_name), index=False)
        print(f"Data saved to {file_name}")

if __name__ == '__main__':
    collect_historical_data()
