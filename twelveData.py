import requests
import configparser

def get_twelve_data(api_key, endpoint, symbol, interval=None):
    url = f'https://api.twelvedata.com/{endpoint}'
    params = {
        'symbol': symbol,
        'apikey': api_key
    }
    
    if interval:
        params['interval'] = interval

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

# APIキーをconfig.iniファイルから読み込む
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config.get('twelvedata', 'api_key')

symbol = 'AAPL'
interval = '1day'

# タイムシリーズデータを取得
time_series_data = get_twelve_data(api_key, 'time_series', symbol, interval)
print(time_series_data)

# 株価データを取得
quote_data = get_twelve_data(api_key, 'quote', symbol)
print(quote_data)

# 指標データを取得
indicator_data = get_twelve_data(api_key, 'rsi', symbol, interval)
print(indicator_data)

# 仮想通貨データを取得
crypto_data = get_twelve_data(api_key, 'time_series', 'BTC/USD', interval)
print(crypto_data)
