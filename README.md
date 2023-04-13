JupyterLab を使ってデータ収集・分析し、売買シグナルを出力する予測用の学習モデルを作成します。

## システム構成
プロジェクトは以下のファイル・ディレクトリ構成で構築されています。


```
.
├── data
│   ├── historical_data.csv
│   └── preprocessed_data.csv
├── models
│   └── lgbm_model.pkl
├── data_collection.py
├── data_preprocessing.py
├── model_training.py
├── model_evaluation.py
└── main.py
```

## 各モジュールの説明
- data_collection.py : bybit から仮想通貨の OHLCV データを取得し、CSV ファイルに保存する。
- data_preprocessing.py : 収集したデータを前処理し、学習用データセットを作成する。
- model_training.py : 前処理済みデータを使って、LightGBM を用いた学習モデルを作成し、モデルを保存する。
- model_evaluation.py : 学習済みモデルの性能を評価し、評価結果を視覚化する。
- main.py : 上記の各ステップを一連のプロセスとして実行する。

## システムの使い方
1. 環境構築
このプロジェクトを実行するためには、Python 3.8 以上が必要です。また、必要なライブラリをインストールするために、以下のコマンドを実行してください。

```
$ python3 -m venv myenv
$ source myenv/bin/activate
(myenv) $ pip install -r requirements.txt
```

2. API キーの設定
data_collection.py で bybit の API キーとシークレットキーを環境変数に設定してください。

```
$ export BYBIT_API_KEY=your_api_key
$ export BYBIT_API_SECRET=your_api_secret
```

3. データ収集
data_collection.py を実行して、bybit から仮想通貨の OHLCV データを取得し、CSV ファイルに保存します。

```
(myenv) $ python3 data_collection.py
```

4. データ前処理
data_preprocessing.py を実行して、収集したデータを前処理し、学習用データセットを作成します。
```
(myenv) $ python3 data_preprocessing.py
```

5. モデル学習
model_training.py を実行して、前処理済みデータを使って、LightGBM を用いた学習モデルを作成し、モデルを保存します。

```
(myenv) $ python3 model_training.py
```

6. モデル評価
model_evaluation.py を実行して、学習済みモデルの性能を評価し、評価結果を視覚化します。

```
(myenv) $ python3 model_evaluation.py
```

7. 一連のプロセスの実行
main.py を実行して、上記の各ステップを一連のプロセスとして実行します。

```
(myenv) $ python3 main.py
```
これで、システムのセットアップと実行が完了です。学習済みモデルを使って、仮想通貨のアルゴリズム取引を行うことができます。