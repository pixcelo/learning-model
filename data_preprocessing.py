import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def generate_features(df):
    # ここで特徴量を生成する（例: 移動平均、ボリンジャーバンド、RSIなど）
    # 今回は単純な例として、終値の5期間移動平均を特徴量とします。
    df['sma5'] = df['close'].rolling(window=5).mean()
    return df

def generate_target(df):
    # 15分後の価格を計算
    df['future_price'] = df['close'].shift(-1)
    
    # 価格が上昇するか下降するかの2値分類を行う
    df['target'] = (df['future_price'] > df['close']).astype(int)
    df.drop(['future_price'], axis=1, inplace=True)
    
    return df

def preprocess_data(csv_file_path):
    # データの読み込み
    df = pd.read_csv(csv_file_path)
    
    # 欠損値の処理（今回は欠損値がないため省略）
    
    # 特徴量の生成
    df = generate_features(df)
    
    # 目的変数の生成
    df = generate_target(df)

    # データの正規化または標準化
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    # 欠損値を含む行を削除
    df.dropna(inplace=True)
    
    # 学習データとテストデータに分割
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    csv_file_name = 'BTCUSDT_15m_20210801_20211231.csv'
    csv_file_path = os.path.join('data', csv_file_name)
    X_train, X_test, y_train, y_test = preprocess_data(csv_file_path)
    print("Data preprocessing completed.")
