実装上の疑問があれば必ず質問してください。

## プロジェクト概要

本リポジトリは、IRT（項目反応理論）の枠組みで実施される CAT（コンピュータ適応型テスト）における項目選択アルゴリズムを研究するリポジトリである。

## 研究の背景と位置づけ

### 元論文

Wang, P., Liu, H., & Xu, M. (2024). An adaptive testing item selection strategy via a deep reinforcement learning approach. *Behavior Research Methods*, 56, 8695–8714. (`papers/s13428-024-02498-x.pdf`)

- IRT に基づく CAT をマルコフ決定過程（MDP）として定式化
- 状態：現在推定されている特性値 θ̂（スカラー）
- DQN を用いて Q 関数を近似し、項目を選択
- シミュレーション・実データの双方で MFI 等5つの従来手法を上回る RMSE・MAE を達成

### 本ディレクトリの目的

元論文における DQN(normal) と MFI のシミュレーション実験を再現すること。

元論文 Table 1（非相関バンク、RMSE）：

| Test length | DQN (normal) | MFI |
|---:|---:|---:|
| 10 | 0.415 | 0.493 |
| 20 | 0.278 | 0.331 |
| 30 | 0.244 | 0.277 |
| 40 | 0.227 | 0.245 |

使用データは非相関バンク（`src/generate_item_banks_3pl.R` で生成した `data/3PL/`、500項目 × 10バンク）。MFI の結果は `data/3PL/results/` に保存。各実験の結果は `EXP***/results/` に保存。

### MDP 定式化の限界と本研究の提案

DQN では状態をスカラー θ̂ のみとするが、①マルコフ性を違反し、②同一推定値でも不確実性が異なるという問題がある。本研究は CAT を POMDP として再定式化し、回答履歴を状態として利用する DRQN（Deep Recurrent Q-Network）を提案手法として採用する。

## 元論文コードの問題点

`Train and test DQN on the simulated banks.py`（プロジェクトルートに配置）は元論文のオリジナルコードで、以下の問題がある：

- パスが `E:/...` の Windows パスにハードコードされており、そのままでは動かない
- `RESPOND` の乱数生成が `np.random.rand(1)`（スカラー1個）のため、全項目が同じ正誤になるバグがある
- `Choose_Action` の ε-greedy 判定に `np.random.randn()`（標準正規分布）を使っており、`np.random.rand()`（一様分布）が正しい
- `action_space = 500` がハードコードされており、バンクサイズと一致しない場合がある

EXP007 以降でこれらを修正済み。

## シミュレーション設定

- IRT モデル：3PLM
- アイテムバンク：500項目（`src/generate_item_banks_3pl.R` で生成）
- パラメータ分布：a ~ N(1.2, 0.25), b ~ N(0, 1), c ~ N(0.25, 0.02)
- バンクの種類：非相関（r_ab = 0）、各10バンク
- 受検者：5,000名、θ ~ N(0, 1)
- テスト長：40問
- 特性値推定：MLE、範囲 [-4, 4]
- 初期特性値：Uniform(-0.5, 0.5)

## 比較手法

1. **MFI** — 最大フィッシャー情報量法（Lord, 1980）
2. **FIWL** — 尤度重み付きフィッシャー情報量法（Veerkamp & Berger, 1997）。1問目は MFI、2問目以降は MLWI。
3. **DQN** — Wang et al. (2024) の手法。状態＝推定特性値（MDP 定式化）
4. **DRQN** — 本研究の提案手法。状態＝回答履歴（POMDP 定式化）

## 評価指標

- Bias
- RMSE
- MAE

（相関係数 r は使用しない）

## 実験一覧

### EXP007 — DQN ベースライン（元論文再現・ローカル動作版）

`EXP007/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

元論文コードのパス問題・バグを修正しローカル/Colab で動くよう整備した実験。アルゴリズム本体は元論文と同一。

主な変更点：
- `E:/...` のハードコードパスを `pathlib.Path` + `find_project_root()` でポータブルに解決
- ハイパーパラメータを `@dataclass Config` に一元化
- `RESPOND` の乱数生成バグを修正（`np.random.rand(1)` → `np.random.random(size=p.shape)`）
- `Choose_Action` の乱数を `np.random.randn()` → `np.random.rand()` に修正
- `action_space` をバンクサイズから自動取得
- ベストモデルを `copy.deepcopy(state_dict)` でメモリ保持し、訓練終了後に保存
- モデル選択基準を RMSE のみに統一
- ステップ別 summary CSV を追加出力

使用バンク：`data/uncorrelated_banks/`（200項目）  
結果保存先：`EXP007/results/`

### EXP008 — TD ターゲット計算の修正

`EXP008/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

EXP007 の replay buffer 学習で Q(next) を計算する際に出題済み項目が候補に含まれてしまう問題を修正。

主な変更点：
- memory に `terminal` フラグと `next_available` マスク（size=action_space）を追加
- Q(next) を `masked_fill(-inf)` で出題済み項目を除外してから argmax
- 終端ステップでは Q_next をゼロにして TD ターゲットを計算

使用バンク：`data/uncorrelated_banks/`（200項目）  
結果保存先：`EXP008/results/`

### EXP009 — data/3PL バンクへの切り替え

`EXP009/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 のアルゴリズムをそのまま維持しつつ、バンクを `data/3PL/`（500項目）に変更した実験。元論文準拠のバンク設定により MFI の RMSE が元論文値に近づいた。

主な変更点：
- バンクディレクトリを `data/3PL/` に固定
- ファイル名を `item_bank_uncor_{id}.csv` に合わせて変更
- `Config.bank_type` を削除（uncor 固定）
- n_items=500

使用バンク：`data/3PL/`（500項目）  
結果保存先：`EXP009/results/`

### EXP010 — RESPOND 修正 + TD ターゲット修正 + data/3PL（最新・これを参照）

`EXP010/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 に対して RESPOND バグ修正・TD ターゲット修正（EXP008 由来）・data/3PL への切り替え（EXP009 由来）を同時に適用した実験。元論文の再現実験として最も修正が揃ったバージョン。

| 変更内容 | EXP007 | EXP008 | EXP009 | EXP010 |
|---|---|---|---|---|
| RESPOND バグ修正 | ✓ | ✓ | ✓ | ✓ |
| TD ターゲットのマスク修正 | — | ✓ | — | ✓ |
| data/3PL 使用 | — | — | ✓ | ✓ |

使用バンク：`data/3PL/`（500項目）  
結果保存先：`EXP010/results/`  
比較用プロット：`EXP010/notebook/plot_rmse_comparison.ipynb`（MFI vs DQN を gamma ごとに別プロット）

## 現状の再現性評価

### MFI の RMSE 比較

| Test length | 元論文 (MFI) | EXP007 (uncor, 200項目) | EXP009/010 (3PL, 500項目) |
|---:|---:|---:|---:|
| 10 | 0.493 | 0.701 | 0.504〜0.522 |
| 20 | 0.331 | 0.462 | 0.327〜0.340 |
| 30 | 0.277 | 0.377 | 0.271〜0.278 |
| 40 | 0.245 | 0.336 | 0.240〜0.245 |

### 現状の問題点

EXP007（data/uncorrelated_banks、200項目）は全ステップで RMSE が元論文より大幅に高い。バンク生成の分散設定が元論文と異なることが原因。

EXP009・EXP010（data/3PL、500項目）は step 40 でほぼ一致するが、序盤（step 10〜20）では依然として乖離が残っている。

## リポジトリ構成

- `Train and test DQN on the simulated banks.py` — 元論文のオリジナル DQN コード（パスハードコード・複数バグあり、EXP007 以降で修正済み）
- `EXP007/` — DQN ベースライン（元論文再現・ローカル動作版）。使用バンク：`data/uncorrelated_banks/`
  - `notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`
  - `results/` — summary CSV・比較プロット
- `EXP008/` — TD ターゲット計算の修正。使用バンク：`data/uncorrelated_banks/`
  - `notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`
  - `results/`
- `EXP009/` — data/3PL バンクへの切り替え（500項目）
  - `notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`
  - `results/`
- `EXP010/` — 全修正統合版（最新）。使用バンク：`data/3PL/`
  - `notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`
  - `notebook/plot_rmse_comparison.ipynb`
  - `results/`
- `src/` — 共有コード
  - `models.py` — DRQN のモデル定義
  - `utils.py` — IRT ユーティリティ（RESPOND, FI, MLE 等）
  - `Functions.R` — R 版の IRT 関数群（Response, MFI, KLP, MLWI, MPWI, MEI, MLE, EAP, MAP を定義）
  - `generate_item_banks.py` — アイテムバンク生成（200項目、`data/uncorrelated_banks/` と `data/correlated_banks/` に出力）
  - `generate_item_banks_3pl.R` — R 版アイテムバンク生成（500項目、`data/3PL/` に出力。元論文再現実験用）
  - `generate_item_banks_2pl.R` — 2PL モデル版アイテムバンク生成
  - `generate_theta_true.py` — 真の特性値生成
  - `IRT-based CAT.R` — IRT ベース CAT シミュレーション（MFI / KLP / MLWI / MPWI / MEI 対応）
  - `Test MFI on the simulated bank.R` — `data/uncorrelated_banks/` を使う MFI 実験（catR パッケージ使用）
  - `Test MFI on the 3PL bank.R` — `data/3PL/` を使う MFI 実験（元論文再現実験用）
  - `Test FIWL on the simulated bank.R` — FIWL 実験スクリプト
  - `split_theta.py` — 特性値の分割ユーティリティ
- `data/` — 生成されたアイテムバンク・特性値データ
  - `uncorrelated_banks/` — 非相関バンク（200項目 × 10バンク）
  - `correlated_banks/` — 有相関バンク（200項目 × 10バンク）
  - `3PL/` — 元論文準拠の非相関バンク（500項目 × 10バンク、`src/generate_item_banks_3pl.R` で生成）
    - `results/` — `src/Test MFI on the 3PL bank.R` の出力先
  - `theta_true/` — 真の特性値
- `seminar_docs/` — ゼミ発表資料・進捗メモ
- `papers/` — 参考論文 PDF・Markdown 版（対応する `.md` がある場合は Markdown を優先して参照）
  - `s13428-024-02498-x.md` — Wang et al. (2024)、元論文
  - `1507.06527v4.md` — Hausknecht & Stone (2015)、DRQN の原典
  - `1704.07978v6.md` — Zhu et al.、ADRQN 系の拡張
  - `363_report.md` — Egorov (2015)、belief vs history ベース入力の比較
  - `IPSJ-GPWS2018034.md` — Oh & 金子 (2018)、DRQN 初期状態学習による改善
  - `s11336-009-9123-2.md` — CD-CAT の基礎論文
  - `10.1177_0013164418790634.md` — Lin & Chang (2019)、制約付き CD-CAT
  - `reinforcement-learning-applied-to-adaptive-classification-5badyfmq0l.md` — Nurakhmetov、適応的分類テストへの RL 導入
- `papers_ja/` — 論文の日本語訳・要約メモ
- `img/` — 図（元論文 Fig.1-2 など）
- `paper.md` — 研究論文の Markdown 版
- `tex_ver1/`, `tex_ver2/`, `tex_ver3/` — 研究論文の LaTeX 版（バージョン管理）
- `refactoring.md` — 本ディレクトリの目的・各実験の説明・再現性評価の記録

## 論文 Markdown の扱い

- 論文を Markdown 化する場合は `papers/<PDFファイル名>.md` の対応を保つ
- 図を切り出す場合は `papers/img/<PDFファイル名>_figN.png` の命名を用いる
- 既存の Markdown 論文を更新する場合は、図リンク切れ、節見出し、表の崩れも確認する

## 実行コマンド

Python の依存関係は `uv` と `pyproject.toml` で管理する。

```bash
uv sync
uv run ruff check -- src
uv run ruff format --check -- src
uv run pyright
```

R スクリプトは以下の形式で実行する。

```bash
Rscript "src/generate_item_banks_3pl.R"
Rscript "src/Test MFI on the 3PL bank.R"
```

ファイル名にスペースが含まれる場合はパスを必ずクォートする。
