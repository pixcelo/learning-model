from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.linear_model import LinearRegression
import numpy as np

class TradingStrategy:
    def __init__(self, commission_rate=0.001, window=3000, threshold=0.05):
        self.COMMISSION_RATE = commission_rate
        self.window = window
        self.threshold = threshold
        self.commission_rate = 0.001

    def prepare_data(self, df):
        # Calculate moving averages
        df['short_ma'] = df['close'].rolling(window=50).mean()
        df['long_ma'] = df['close'].rolling(window=200).mean()

        # Determine the trend using linear regression
        df['trend'] = 0
        for i in range(self.window - 1, len(df)):
            y = df['close'].iloc[i - self.window + 1:i + 1].values.reshape(-1, 1)
            X = np.arange(self.window).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0][0]

            # Threshold for determining trend
            if slope > self.threshold:
                df.loc[i, 'trend'] = 1
            elif slope < -self.threshold:
                df.loc[i, 'trend'] = -1

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
        rsi = df.loc[i, 'RSI']
        trend = df.loc[i, 'trend']

        if portfolio['position'] == 'long' and rsi > 70:
            return 'exit_long'
        elif portfolio['position'] == 'short' and rsi < 20:
            return 'exit_short'
        elif rsi < 20 and trend == -1:
            return 'entry_long'
        elif rsi > 70 and trend == 1:
            return 'entry_short'
        else:
            return None
        
    def trade_conditions2(self, df, i, portfolio): 
        rsi = df.loc[i, 'RSI']
        trend = df.loc[i, 'trend']

        if portfolio['position'] == 'long' and rsi > 70:
            return 'exit_long'
        elif portfolio['position'] == 'short' and rsi < 20:
            return 'exit_short'
        elif rsi < 20 and trend == 1: # 売られすぎた後にトレンドが変換したと仮定
            return 'entry_long'
        elif rsi > 70 and trend == -1: # 買われすぎた後にトレンドが変換したと仮定
            return 'entry_short'
        else:
            return None
        
    def trade_conditions3(self, df, i, portfolio): 
        rsi = df.loc[i, 'RSI']
        trend = df.loc[i, 'trend']
        close = df.loc[i, 'close']

        # 利確と損切りの閾値
        TAKE_PROFIT = 10
        STOP_LOSS = -10

        if portfolio['position'] == 'long':
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_long'
        elif rsi < 20 and trend == 1:
            return 'entry_long'
        else:
            return None
        
    def trade_conditions4(self, df, i, portfolio): 
        rsi = df.loc[i, 'RSI']
        trend = df.loc[i, 'trend']
        close = df.loc[i, 'close']

        # 利確と損切りの閾値
        TAKE_PROFIT = 10
        STOP_LOSS = -10

        if portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif rsi > 60 and trend == -1:
            return 'entry_short'
        else:
            return None
        
    # close が 200MAを上抜けたら　追随してlong (ATR利確・損切り)
    def trade_conditions5(self, df, i, portfolio): 
        atr = df.loc[i, 'ATR']
        long_ma = df.loc[i, 'long_ma']
        close = df.loc[i, 'close']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_long_ma = df.loc[i - 1, 'long_ma'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 2
        STOP_LOSS = atr * -1

        if portfolio['position'] == 'long':
            profit = (close - portfolio['entry_price']) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_long'
        elif prev_close is not None and prev_long_ma is not None \
            and prev_close < prev_long_ma and close > long_ma:
            return 'entry_long'
        else:
            return None
        
    # close が 200MAを下抜けたら　追随してshort (ATR利確・損切り)
    def trade_conditions6(self, df, i, portfolio): 
        atr = df.loc[i, 'ATR']
        long_ma = df.loc[i, 'long_ma']
        close = df.loc[i, 'close']

        prev_close = df.loc[i - 1, 'close'] if i > 0 else None
        prev_long_ma = df.loc[i - 1, 'long_ma'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 2
        STOP_LOSS = atr * -1

        if portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif prev_close is not None and prev_long_ma is not None \
            and prev_close > prev_long_ma and close < long_ma:
            return 'entry_short'
        else:
            return None
        
    # RSIの50を基点にトレード
    def trade_conditions7(self, df, i, portfolio): 
        atr = df.loc[i, 'ATR']
        rsi = df.loc[i, 'RSI']
        close = df.loc[i, 'close']

        prev_rsi = df.loc[i - 1, 'RSI'] if i > 0 else None

        # 利確と損切りの閾値
        TAKE_PROFIT = atr * 2
        STOP_LOSS = atr * -1

        if portfolio['position'] == 'short':
            profit = (portfolio['entry_price'] - close) * (1 - self.commission_rate)
            if profit > TAKE_PROFIT or profit < STOP_LOSS:
                return 'exit_short'
        elif prev_rsi is not None and 50 > prev_rsi and 50 < rsi:
            return 'entry_short'
        else:
            return None

