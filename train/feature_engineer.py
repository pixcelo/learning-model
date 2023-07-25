import numpy as np
import talib

class FeatureEngineer:
    def feature_engineering(self, df):
        open = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        hilo = (high + low) / 2

        df['RSI_ST'] = talib.RSI(close)
        # df['RSI_LOG'] = log_transform_feature(talib.RSI(close))
        df['MACD'], _, _ = talib.MACD(close)
        df['MACD_ST'], _, _ = talib.MACD(close)
        df['ATR'] = talib.ATR(high, low, close)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        
        df['SMA10'] = talib.SMA(close, timeperiod=10)
        df['SMA50'] = talib.SMA(close, timeperiod=50)
        df['SMA200'] = talib.SMA(close, timeperiod=200)
        
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(close)
        df['BBANDS_upperband'] = (df['BB_UPPER'] - hilo)
        df['BBANDS_middleband'] = (df['BB_MIDDLE'] - hilo)
        df['BBANDS_lowerband'] = (df['BB_LOWER'] - hilo)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)/close
        df['MON'] = talib.MOM(close, timeperiod=5)
        df['OBV'] = talib.OBV(close, volume)
        # df['CCI'] = talib.CCI(close, high, low, timeperiod=14)

        # Calculate high_close_comparison
        df['High_Close_Comparison'] = self.calculate_high_close_comparison(df)
        df['consecutive_up'], df['consecutive_down']  = self.calculate_consecutive_candles(df)
        df['double_top'], df['double_bottom'] = self.detect_double_top_bottom(df)

        # Add triangle pattern feature
        df = self.detect_triangle_pattern(df)

        # add features 20230508
        # df = stochastic_crossover(df)
        # df = macd_divergence(df)
        # df = pin_bar_pattern(df)
        # df = triple_top_bottom_pattern(df)
        # df = line_touch_bounce_both_sides(df)
        df = self.parallel_channel(df)
        # df = pivot_points(df)
        # df = fibonacci_retracement_levels(df)
        # df = triple_barrier(df)
        # df = bollinger_band_touch(df)

        df = df.dropna()
        df = df.reset_index(drop=True)

        return df

    def log_transform_feature(self, X):
        X[X <= 0] = np.finfo(float).eps
        return np.log(X)

    def support_resistance(self, df, window=20):
        high = df['high']
        low = df['low']
        close = df['close']
        df['support'] = low.rolling(window=window, min_periods=1).min()
        df['resistance'] = high.rolling(window=window, min_periods=1).max()
        return df

    def calculate_high_close_comparison(self, df):
        high = df['high'].values
        close = df['close'].values
        higher_high = np.zeros(len(high), dtype=int)
        higher_close = np.zeros(len(close), dtype=int)
        higher_high[1:] = high[1:] > high[:-1]
        higher_close[1:] = close[1:] > close[:-1]
        high_close_comparison = higher_high & higher_close
        return high_close_comparison

    def calculate_consecutive_candles(self, df):
        close = df['close'].values

        consecutive_up = np.zeros_like(close, dtype=int)
        consecutive_down = np.zeros_like(close, dtype=int)

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                consecutive_up[i] = consecutive_up[i - 1] + 1
                consecutive_down[i] = 0
            elif close[i] < close[i - 1]:
                consecutive_up[i] = 0
                consecutive_down[i] = consecutive_down[i - 1] + 1
            else:
                consecutive_up[i] = 0
                consecutive_down[i] = 0

        return consecutive_up, consecutive_down

    def detect_double_top_bottom(self, df, window=5, tolerance=0.03):
        double_top = np.zeros(len(df), dtype=int)
        double_bottom = np.zeros(len(df), dtype=int)

        close = df['close'].values
        close_ext = np.pad(close, (window, window), mode='edge')

        for i in range(window, len(df) - window):
            considered_range = close_ext[i:i + window * 2 + 1]
            max_index = np.argmax(considered_range)
            min_index = np.argmin(considered_range)

            if max_index == window:
                max_left = np.max(considered_range[:window])
                max_right = np.max(considered_range[window + 1:])
                max_avg = (max_left + max_right) / 2

                if np.abs(considered_range[window] - max_avg) / considered_range[window] <= tolerance:
                    double_top[i] = 1

            if min_index == window:
                min_left = np.min(considered_range[:window])
                min_right = np.min(considered_range[window + 1:])
                min_avg = (min_left + min_right) / 2

                if np.abs(considered_range[window] - min_avg) / considered_range[window] <= tolerance:
                    double_bottom[i] = 1

        return double_top, double_bottom

    def detect_triangle_pattern(self, df, window=20):
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate ascending trendline
        df['ascending_trendline'] = (
            low.rolling(window=window, min_periods=1).min()
            + (high.rolling(window=window, min_periods=1).max()
            - low.rolling(window=window, min_periods=1).min()) * np.arange(1, len(df) + 1) / window
        )

        # Calculate descending trendline
        df['descending_trendline'] = (
            high.rolling(window=window, min_periods=1).max()
            - (high.rolling(window=window, min_periods=1).max()
            - low.rolling(window=window, min_periods=1).min()) * np.arange(1, len(df) + 1) / window
        )

        # Check if close price is between the trendlines
        df['triangle_pattern'] = np.where(
            (close > df['ascending_trendline']) 
            & (close < df['descending_trendline']), 1, 0
        )

        return df

    def stochastic_crossover(self, df, k_period=14, d_period=3, overbought_level=80, oversold_level=20):
        high = df['high']
        low = df['low']
        close = df['close']
        # Calculate the Stochastic Oscillator
        k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        # Find the crossover points
        buy_signal = np.where((k.shift(1) < d.shift(1)) & (k > d) & (k <= oversold_level), 1, 0)
        sell_signal = np.where((k.shift(1) > d.shift(1)) & (k < d) & (k >= overbought_level), 1, 0)
        # Add the features to the DataFrame
        df['stoch_buy_signal'] = buy_signal
        df['stoch_sell_signal'] = sell_signal
        return df

    def macd_divergence(self, df, fast_period=12, slow_period=26, signal_period=9):
        close = df['close']
        
        # Calculate MACD
        macd, signal, _ = talib.MACD(close, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        
        # Calculate the difference between MACD and its signal line
        macd_diff = macd - signal
        
        # Find divergences
        price_higher_high = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
        price_lower_low = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
        
        macd_diff_lower_high = (macd_diff < macd_diff.shift(1)) & (macd_diff.shift(1) < macd_diff.shift(2))
        macd_diff_higher_low = (macd_diff > macd_diff.shift(1)) & (macd_diff.shift(1) > macd_diff.shift(2))
        
        bullish_divergence = price_lower_low & macd_diff_higher_low
        bearish_divergence = price_higher_high & macd_diff_lower_high
        
        # Add the features to the DataFrame
        df['bullish_macd_divergence'] = bullish_divergence.astype(int)
        df['bearish_macd_divergence'] = bearish_divergence.astype(int)
        
        return df

    def pin_bar_pattern(self, df):
        high = df['high']
        low = df['low']
        open_ = df['open']
        close = df['close']

        # Define the pin bar pattern
        nose_proportion = 0.6
        body = abs(close - open_)
        wick_upper = np.where(close < open_, high - close, high - open_)
        wick_lower = np.where(close < open_, open_ - low, close - low)

        bullish_pin_bar = (wick_lower > body * nose_proportion) & (wick_upper < body)
        bearish_pin_bar = (wick_upper > body * nose_proportion) & (wick_lower < body)

        # Add the features to the DataFrame
        df['bullish_pin_bar'] = bullish_pin_bar.astype(int)
        df['bearish_pin_bar'] = bearish_pin_bar.astype(int)

        return df

    def triple_top_bottom_pattern(self, df):
        high = df['high']
        low = df['low']
        close = df['close']

        # Define the pattern parameters
        n = 3  # Number of tops/bottoms
        threshold = 0.03  # Price tolerance

        # Identify tops and bottoms
        tops = high.rolling(window=3).apply(lambda x: (x[1] > x[0]) & (x[1] > x[2]), raw=True).fillna(0).astype(int)
        bottoms = low.rolling(window=3).apply(lambda x: (x[1] < x[0]) & (x[1] < x[2]), raw=True).fillna(0).astype(int)

        # Check for triple top and bottom patterns
        triple_top = (tops == 1).rolling(window=n).sum() == n
        triple_bottom = (bottoms == 1).rolling(window=n).sum() == n

        # Check price similarity for triple top and bottom patterns
        triple_top_pattern = triple_top & (high.rolling(window=n).std() / high.rolling(window=n).mean() < threshold)
        triple_bottom_pattern = triple_bottom & (low.rolling(window=n).std() / low.rolling(window=n).mean() < threshold)

        # Add the features to the DataFrame
        df['triple_top_pattern'] = triple_top_pattern.astype(int)
        df['triple_bottom_pattern'] = triple_bottom_pattern.astype(int)

        return df

    def line_touch_bounce_both_sides(self, df, window=20, tolerance=0.03):
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate support and resistance
        df = self.support_resistance(df, window)

        support = df['support']
        resistance = df['resistance']

        # Check if the price is close to the support line
        close_to_support = abs(close - support) <= (tolerance * close)

        # Check if the price is close to the resistance line
        close_to_resistance = abs(close - resistance) <= (tolerance * close)

        # Check if the price bounces from the support line
        bounce_from_support = (close_to_support.shift(1)) & (close > close.shift(1))

        # Check if the price bounces from the resistance line
        bounce_from_resistance = (close_to_resistance.shift(1)) & (close < close.shift(1))

        # Add the features to the DataFrame
        df['bounce_from_support'] = bounce_from_support.astype(int)
        # df['bounce_from_resistance'] = bounce_from_resistance.astype(int)

        return df

    def parallel_channel(df, window=20, tolerance=0.03):
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate the moving averages for the high and low prices
        high_mavg = high.rolling(window=window).mean()
        low_mavg = low.rolling(window=window).mean()

        # Calculate the channel's upper and lower boundaries
        channel_upper = high_mavg + (high_mavg - low_mavg) * tolerance
        channel_lower = low_mavg - (high_mavg - low_mavg) * tolerance

        # Add the channel boundaries to the DataFrame
        df['channel_upper'] = channel_upper
        df['channel_lower'] = channel_lower

        # Check if the price is close to the channel boundaries
        close_to_upper = abs(close - channel_upper) <= (tolerance * close)
        close_to_lower = abs(close - channel_lower) <= (tolerance * close)

        # Check if the price bounces from the channel boundaries
        bounce_from_upper = (close_to_upper.shift(1)) & (close < close.shift(1))
        bounce_from_lower = (close_to_lower.shift(1)) & (close > close.shift(1))

        # Add the bounce features to the DataFrame
        df['bounce_from_channel_upper'] = bounce_from_upper.astype(int)
        df['bounce_from_channel_lower'] = bounce_from_lower.astype(int)

        return df

    def pivot_points(self, df):
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate the pivot point using the high, low, and close prices
        pivot_point = (high + low + close) / 3

        # Calculate support and resistance levels
        support1 = (2 * pivot_point) - high
        resistance1 = (2 * pivot_point) - low
        support2 = pivot_point - (high - low)
        resistance2 = pivot_point + (high - low)

        # Add pivot point, support, and resistance levels to the DataFrame
        df['pivot_point'] = pivot_point
        df['support1'] = support1
        df['resistance1'] = resistance1
        df['support2'] = support2
        df['resistance2'] = resistance2

        return df


    def fibonacci_retracement_levels(df, window=20):
        high = df['high'].rolling(window=window, min_periods=1).max()
        low = df['low'].rolling(window=window, min_periods=1).min()

        # Calculate the range
        price_range = high - low

        # Define Fibonacci levels
        levels = [1.618, 2.0, 2.618, 3.0, 3.5, 3.618, 4.236, 5.5]

        # Calculate retracement levels
        for level in levels:
            retracement = price_range * level
            support = high - retracement
            resistance = low + retracement

            # Add support and resistance levels to the DataFrame
            df['fib_support_{level}'] = support
            df['fib_resistance_{level}'] = resistance

        return df

    def triple_barrier(df, profit_take=0.03, stop_loss=0.03, time_horizon=20):
        close = df['close']
        price_diff = close.pct_change()

        # Calculate profit take and stop loss signals
        profit_take_signal = price_diff > profit_take
        stop_loss_signal = price_diff < -stop_loss

        # Calculate time barrier signal
        time_barrier_signal = close.index.to_series().diff(periods=time_horizon) == 0

        # Add the signals to the DataFrame
        df['triple_barrier_profit_take'] = profit_take_signal.astype(int)
        df['triple_barrier_stop_loss'] = stop_loss_signal.astype(int)
        df['triple_barrier_time'] = time_barrier_signal.astype(int)

        return df

    def bollinger_band_touch(df):
        close = df['close']
        
        # Calculate the Bollinger Bands
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        # Check if the price touches the Bollinger Bands
        df['touch_upper_band'] = (close >= upper_band).astype(int)
        df['touch_lower_band'] = (close <= lower_band).astype(int)

        return df
