import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from joblib import dump
from data_preprocessing import preprocess_data

def train_model(X_train, y_train):
    # LightGBMモデルの選択
    model = lgb.LGBMClassifier(random_state=42)

    # ハイパーパラメータの設定
    param_grid = {
        'num_leaves': [31, 127],
        'min_data_in_leaf': [30, 50, 100],
        'learning_rate': [0.01, 0.1, 0.5]
    }

    # GridSearchCVを用いたハイパーパラメータチューニング
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # 最適なハイパーパラメータを表示
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best score: {grid.best_score_}")

    # 最適なハイパーパラメータで学習したモデルを返す
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    # テストデータを用いたモデルの評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, model_file_path):
    # 学習済みモデルの保存
    dump(model, model_file_path)

if __name__ == '__main__':
    # 前処理済みデータの取得
    csv_file_name = 'BTCUSDT_15m_20210801_20211231.csv'
    csv_file_path = os.path.join('data', csv_file_name)
    X_train, X_test, y_train, y_test = preprocess_data(csv_file_path)

    # モデルの学習
    model = train_model(X_train, y_train)

    # モデルの評価
    evaluate_model(model, X_test, y_test)

    # 学習済みモデルの保存
    model_file_path = os.path.join('models', 'lightgbm_model.joblib')
    save_model(model, model_file_path)
