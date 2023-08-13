from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import numpy as np

class TradingStrategy:
    def __init__(self, commission_rate=0.001, window=3000, threshold=0.05):
        self.COMMISSION_RATE = commission_rate
        self.window = window
        self.threshold = threshold
        self.commission_rate = 0.001

    def prepare_data(self, df):
        # Calculate moving averages
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['short_ma'] = df['close'].rolling(window=50).mean()
        df['long_ma'] = df['close'].rolling(window=200).mean()

        df.loc[df['short_ma'] > df['long_ma'], 'trend'] = 1
        df.loc[df['short_ma'] < df['long_ma'], 'trend'] = 0

        rsi_indicator = RSIIndicator(close=df['close'])
        df['RSI'] = rsi_indicator.rsi()
        average_true_range = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['ATR'] = average_true_range.average_true_range()

        return df

    # trade logic
    def trade_conditions1(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 1
        STOP_LOSS = atr * -1

        if portfolio['position'] == 'long':
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        else:
            return None
        
    
    def trade_conditions2(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 1
        STOP_LOSS = atr * -1

        if portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None
        
    def trade_conditions3(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 2
        STOP_LOSS = atr * -1

        if portfolio['position'] == 'long':
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_long'
        elif portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None
        
    def trade_conditions4(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 1
        STOP_LOSS = atr * -1

        if portfolio['position'] == 'long':
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_long'
        elif portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None
        
    def trade_conditions5(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 1
        STOP_LOSS = atr * -2

        if portfolio['position'] == 'long':
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_long'
        elif portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None
        

    def trade_conditions6(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 1.5
        STOP_LOSS = atr * -1.5

        if portfolio['position'] == 'long':
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_long'
        elif portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None
        
    def trade_conditions7(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 1
        STOP_LOSS = atr * -1

        # トレーリングストップのロジック
        if portfolio['position'] == 'long':
            portfolio['trailing_stop'] = max(portfolio['trailing_stop'], close - STOP_LOSS) if 'trailing_stop' in portfolio else close - STOP_LOSS
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or close < portfolio['trailing_stop']:
                return 'exit_long'
        elif portfolio['position'] == 'short':
            portfolio['trailing_stop'] = min(portfolio['trailing_stop'], close + STOP_LOSS) if 'trailing_stop' in portfolio else close + STOP_LOSS
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or close > portfolio['trailing_stop']:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None

    # ストップロスのみ
    def trade_conditions8(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        STOP_LOSS = atr * -1

        # トレーリングストップのロジック
        if portfolio['position'] == 'long':
            portfolio['trailing_stop'] = max(portfolio['trailing_stop'], close - STOP_LOSS) if 'trailing_stop' in portfolio else close - STOP_LOSS
            if close < portfolio['trailing_stop']:
                return 'exit_long'
        elif portfolio['position'] == 'short':
            portfolio['trailing_stop'] = min(portfolio['trailing_stop'], close + STOP_LOSS) if 'trailing_stop' in portfolio else close + STOP_LOSS
            if close > portfolio['trailing_stop']:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None
        

    def trade_conditions9(self, df, i, portfolio):
        atr = df.loc[i, 'ATR']
        close = df.loc[i, 'close']
        ma = df.loc[i, 'SMA20']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_ma = df.loc[i - 1, 'SMA20'] if i > 0 else None

        # 利確と損切りの閾値
        STOP_LOSS = atr * -0.5

        # トレーリングストップのロジック
        if portfolio['position'] == 'long':
            portfolio['trailing_stop'] = max(portfolio['trailing_stop'], close - STOP_LOSS) if 'trailing_stop' in portfolio else close - STOP_LOSS
            if close < portfolio['trailing_stop']:
                return 'exit_long'
        elif portfolio['position'] == 'short':
            portfolio['trailing_stop'] = min(portfolio['trailing_stop'], close + STOP_LOSS) if 'trailing_stop' in portfolio else close + STOP_LOSS
            if close > portfolio['trailing_stop']:
                return 'exit_short'
        elif prev_close is not None and prev_ma is not None \
            and prev_close < prev_ma and close > ma:
            return 'entry_long'
        elif prev_close is not None and prev_ma is not None \
            and prev_close > prev_ma and close < ma:
            return 'entry_short'
        else:
            return None