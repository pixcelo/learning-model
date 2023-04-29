import os
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

api_key = config.get('bybit', 'api_key')
secret_key = config.get('bybit', 'secret_key')

exchange = ccxt.bybit({
    'apiKey': api_key,
    'secret': secret_key,
})

def add_prefix_to_columns(df, prefix):
    df.columns = [f'{prefix}_{column}' for column in df.columns]
    return df

def fetch_order_book(exchange, symbol, timeframe, since):
    fetched_data = exchange.fetch_order_book(symbol)
    timestamp = int(time.time() * 1000)
    bids_data = [[timestamp, price, amount] for price, amount in fetched_data['bids']]
    asks_data = [[timestamp, price, amount] for price, amount in fetched_data['asks']]
    time.sleep(exchange.rateLimit / 1000 * 2)
    return bids_data, asks_data

def fetch_order_book_multiple_timeframes(exchange, symbol, timeframes, start_date, end_date):
    data = {}
    for timeframe in timeframes:
        print(f"Fetching {symbol} {timeframe} order book data...")
        data[timeframe] = {'bids': [], 'asks': []}
        current_date = start_date
        while current_date < end_date:
            bids_data, asks_data = fetch_order_book(exchange, symbol, timeframe, since=current_date)
            data[timeframe]['bids'].extend(bids_data)
            data[timeframe]['asks'].extend(asks_data)
            current_date = current_date + timedelta(milliseconds=exchange.parse_timeframe(timeframe) * 1000)
    print(f"{symbol} {timeframe} order book data fetched.")
    return data

def to_unix_timestamp(date):
    return int(date.timestamp() * 1000)

def save_order_book_to_csv(data, file_name):
    df = pd.DataFrame(data)
    column_names = ['timestamp', 'price', 'amount']
    df.columns = column_names
    file_path = os.path.join("data", file_name)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def collect_order_book_data():
    symbol = 'BTC/USDT'
    timeframes = ['15m', '1h', '4h']
    start_date = datetime(2021, 8, 1)
    end_date = datetime(2022, 3, 31)
    unix_start_date = to_unix_timestamp(start_date)
    unix_end_date = to_unix_timestamp(end_date)

    order_book_data = fetch_order_book_multiple_timeframes(exchange, symbol, timeframes, unix_start_date, unix_end_date)

    for timeframe in timeframes:
        for side in ['bids', 'asks']:
            file_name = f'orderbook_{side}_{timeframe}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
            save_order_book_to_csv(order_book_data[timeframe][side], file_name)

if __name__ == '__main__':
    collect_order_book_data()

