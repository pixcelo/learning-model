import os
import pandas as pd
import numpy as np
import talib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# データの読み込み
def load_data(file_names):
    dfs = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        dfs.append(df)
    return dfs

# 特徴量エンジニアリング
def feature_engineering(df):
    open = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values

    # TA-Libを使用して一般的なテクニカル指標を計算
    df['RSI'] = talib.RSI(close)
    df['MACD'], _, _ = talib.MACD(close)
    df['ATR'] = talib.ATR(high, low, close)
    df['ADX'] = talib.ADX(high, low, close)
    df['SMA'] = talib.SMA(close)
    df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(close)

    # 欠損値の削除
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df

# ラベルデータ作成
def create_label(df, lookahead=1):
    df['target'] = (df['close'].shift(-lookahead) > df['close']).astype(int)
    df = df.dropna()
    return df

# 学習と評価
def train_and_evaluate(df):
    features = df.drop('target', axis=1)
    labels = df['target']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[train_data, test_data],
        num_boost_round=10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
        callbacks=[lgb.early_stopping(stopping_rounds=10, 
                verbose=True), # early_stopping用コールバック関数
                lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数
    )

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    file_names = [
        "data/BTCUSD_15m_20210801_20211231.csv", 
        "data/BTCUSD_1h_20210801_20211231.csv", 
        "data/BTCUSD_4h_20210801_20211231.csv"]
    dfs = load_data(file_names)

    # 各タイムフレームのデータに対して特徴量エンジニアリングとラベル作成を行う
    processed_dfs = []
    for df in dfs:
        processed_df = feature_engineering(df)
        processed_df = create_label(processed_df)
        processed_dfs.append(processed_df)

    # 複数のタイムフレームのデータを結合（インデックスが一致するように注意）
    combined_df = pd.concat(processed_dfs, axis=1).dropna()
    print(combined_df)

    # モデルの学習と評価を行う
    model = train_and_evaluate(combined_df)
 
    # モデルを保存する
    model_path = os.path.join("model", "model.pkl")
    joblib.dump(model, model_path)