import requests

def get_ticker(coin):
    url = "https://api.bybit.com//v5/account/fee-rate"
    params = {
        'category': 'linear',
        'baseCoin': coin
        }
    response = requests.get(url, params=params)
    data = response.json()
    return data

ticker_data = get_ticker('BTC')
print(ticker_data)
