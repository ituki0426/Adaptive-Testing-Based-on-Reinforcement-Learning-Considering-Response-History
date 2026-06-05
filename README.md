# Grad_Research

このリポジトリは、CAT/CD-CAT 関連の論文管理、和訳メモ、ゼミ資料、実験コードをまとめるための作業ディレクトリです。

## ディレクトリ構成

- `papers/`: 論文 PDF
- `papers_ja/`: 論文の日本語訳・要約 `.md`
- `seminar_docs/`: ゼミ用 `.md` 資料
- `src/`: 実験コード（R / Python）
- `img/`: 図・スクリーンショット

## 使い方（運用ルール）

- 新しい論文 PDF は `papers/` に配置
- 論文の和訳・要約 `.md` は `papers_ja/` に配置
- ゼミで使う資料 `.md` は `seminar_docs/` に配置
- コードは `src/` に配置

## 実行メモ

- R スクリプト実行例:
  - `Rscript src/IRT-based CAT.R`
  - `Rscript src/Functions.R`
- Python スクリプト実行例:
  - `uv run python "src/Train and test DQN on the simulated banks.py"`
  - `uv run python "src/Train and test DQN on the real responses.py"`

## Python 環境管理

Python の依存関係は `uv` と `pyproject.toml` で管理します。

```powershell
# 仮想環境を作成し、依存関係を同期する
uv sync

# 開発用ツールも含めて同期する
uv sync --group dev
```

Python バージョンは `.python-version` で `3.11.8` に固定しています。
依存パッケージを追加する場合は、次のように `uv add` を使います。

```powershell
uv add パッケージ名
uv add --dev パッケージ名
```

## Ruff の使い方

Python コードの lint、修正、format には Ruff を使います。
ファイル名にスペースが含まれる場合は、パスを必ずダブルクォートで囲んでください。

### 通常の Ruff コマンド

```powershell
# 問題をチェックする
uv run ruff check -- "src\Train and test DQN on the real responses.py"

# Ruff が自動修正できる問題を修正する
uv run ruff check --fix -- "src\Train and test DQN on the real responses.py"

# コードを整形する
uv run ruff format -- "src\Train and test DQN on the real responses.py"

# 整形が必要かだけ確認する
uv run ruff format --check -- "src\Train and test DQN on the real responses.py"
```

`check --fix` は lint 由来の自動修正、`format` はコード整形です。両方を適用したい場合は、次の順に実行します。

```powershell
uv run ruff check --fix -- "src\Train and test DQN on the real responses.py"
uv run ruff format -- "src\Train and test DQN on the real responses.py"
```

## 補足

- `tmp_*.txt` は作業中の一時ファイルです。必要に応じて削除してください。

## 各ブランチでの実験結果

mainブランチ

DQN
| checkpoint       |       Bias |      RMSE |       MAE |
| ---------------- | ---------: | --------: | --------: |
| subject 200      |     −0.052 |     0.310 |     0.205 |
| subject 400      |     −0.008 |     0.240 |     0.183 |
| subject 600      |     −0.025 |     0.259 |     0.179 |
| subject 800      |     −0.006 |     0.266 |     0.183 |
| **subject 1000** | **−0.022** | **0.220** | **0.165** |
| subject 1200     |     −0.053 |     0.292 |     0.189 |
| subject 1400     |     −0.026 |     0.235 |     0.166 |
| subject 1600     |     −0.010 |     0.298 |     0.188 |
| subject 1800     |     −0.026 |     0.230 |     0.161 |

DDRQN
| checkpoint | Bias | RMSE | MAE |
|---|---:|---:|---:|
| subject 200 | −0.061 | 0.304 | 0.181 |
| **subject 400** | **−0.020** | **0.204** | **0.143** |
| subject 600 | −0.021 | 0.228 | 0.160 |
| subject 800 | −0.019 | 0.205 | 0.150 |
| subject 1000 | −0.029 | 0.255 | 0.168 |
| subject 1200 | −0.010 | 0.218 | 0.158 |
| subject 1400 | −0.012 | 0.306 | 0.184 |
| subject 1600 | −0.024 | 0.229 | 0.151 |
| subject 1800 | −0.012 | 0.241 | 0.164 |

DBQN

| checkpoint       |      Bias |      RMSE |       MAE |
| ---------------- | --------: | --------: | --------: |
| subject 200      |     0.002 |     0.233 |     0.179 |
| subject 400      |     0.007 |     0.242 |     0.185 |
| subject 600      |     0.010 |     0.240 |     0.189 |
| subject 800      |     0.030 |     0.231 |     0.178 |
| subject 1000     |    −0.002 |     0.237 |     0.182 |
| **subject 1200** | **0.010** | **0.196** | **0.156** |
| subject 1400     |    −0.002 |     0.219 |     0.170 |
| subject 1600     |     0.007 |     0.215 |     0.163 |
| subject 1800     |    −0.002 |     0.202 |     0.156 |


### 2.v1     

既存の LSTM は 1 個前の時系列データのみを入力していたが、これを修正して `TIME_STEP` 分だけ LSTM に入力するようにしたブランチ。mainブランチとは異なり、v1ブランチではシミュレーションデータを利用して、各モデルを学習・評価している。

DQN