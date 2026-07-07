本ディレクトリの目的

元論文におけるDQN(normal)とMFIのシミュレーション実験を再現すること。

元論文のテーブル1

| Test length | DQN (normal) | MFI |
|---:|---:|---:|
| 10 | 0.415 | 0.493 |
| 20 | 0.278 | 0.331 |
| 30 | 0.244 | 0.277 |
| 40 | 0.227 | 0.245 |

使うデータセットは非相関データセット。src/generate_item_banks_3pl.Rによって生成されたdata/3PL/item_bank_uncor_1.csv~data/3PL/item_bank_uncor_10.csvを使用。

MFIの結果はdata/3PL/results内に保存。

各実験の結果はEXP(実験番号)/results内に保存。

元論文のDQNのプログラムはTrain and test DQN on the simulated banks.py

## 元論文のプログラム（Train and test DQN on the simulated banks.py）

プロジェクトルートに置かれた元論文のオリジナルコード。以下の問題点がある：

- パスが `E:/...` の Windows パスにハードコードされており、そのままでは動かない
- `RESPOND` の乱数生成が `np.random.rand(1)`（スカラー1個）のため、全項目が同じ正誤になるバグがある
- `Choose_Action` の ε-greedy 判定に `np.random.randn()`（標準正規分布）を使っており、`np.random.rand()`（一様分布）が正しい
- `action_space = 500` がハードコードされており、バンクサイズと一致しない場合がある

## 各実験の説明

### EXP007

**DQN ベースライン（元論文再現・ローカル動作版）**

`EXP007/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

元論文コードをローカル環境・Colab で動くよう整備した実験。アルゴリズム本体は元論文と同一。

元論文コードからの主な変更点：
- `E:/...` のハードコードパスを `pathlib.Path` + `find_project_root()` でポータブルに解決
- ハイパーパラメータを `@dataclass Config` に一元化
- `RESPOND` の乱数生成バグを修正（`np.random.rand(1)` → `np.random.random(size=p.shape)`）
- `Choose_Action` の乱数を `np.random.randn()` → `np.random.rand()` に修正
- `action_space` をバンクサイズから自動取得
- ベストモデルを `copy.deepcopy(state_dict)` でメモリ保持し、訓練終了後に保存
- モデル選択基準を RMSE のみに統一
- ステップ別 summary CSV を追加出力

使用バンク：`data/uncorrelated_banks/`（n_items=500 設定、実効項目数はバンクサイズに依存）
結果保存先：`EXP007/results/`
詳細：`EXP007/exp_summary.md`

### EXP008

**TD ターゲット計算の修正**

`EXP008/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

EXP007 の replay buffer 学習で Q(next) を計算する際に出題済み項目が候補に含まれてしまう問題を修正。

主な変更点：
- memory に `terminal` フラグと `next_available` マスク（size=action_space）を追加
- Q(next) を `masked_fill(-inf)` で出題済み項目を除外してから argmax
- 終端ステップでは Q_next をゼロにして TD ターゲットを計算

使用バンク：`data/uncorrelated_banks/`
結果保存先：`EXP007/results/`（EXP007 と共用）
詳細：`EXP008/exp_summary.md`

### EXP009

**data/3PL バンクへの切り替え**

`EXP009/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 のアルゴリズムをそのまま維持しつつ、バンクを `data/3PL/`（500項目）に変更した実験。

主な変更点：
- バンクディレクトリを `data/3PL/` に固定
- ファイル名を `item_bank_uncor_{id}.csv` に合わせて変更
- `Config.bank_type` を削除（uncor 固定）
- n_items=500

使用バンク：`data/3PL/`（`src/generate_item_banks_3pl.R` で生成、500項目）
結果保存先：`EXP009/results/`
詳細：`EXP009/exp_summary.md`

### EXP010

**RESPOND 修正 + TD ターゲット修正 + data/3PL**

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
詳細：`EXP010/exp_summary.md`

## 現状の再現性評価

### MFI の RMSE 比較

元論文 Table 1 の MFI RMSE と、各実験での実測値：

| Test length | 元論文 (MFI) | EXP007 (uncor, 200項目) | EXP009/010 (3PL, 500項目) |
|---:|---:|---:|---:|
| 10 | 0.493 | 0.701 | 0.504〜0.522 |
| 20 | 0.331 | 0.462 | 0.327〜0.340 |
| 30 | 0.277 | 0.377 | 0.271〜0.278 |
| 40 | 0.245 | 0.336 | 0.240〜0.245 |

### 現状の問題点

**EXP007（data/uncorrelated_banks、200項目）**
- 全ステップで RMSE が元論文より大幅に高い（step 10 で 0.701 vs 0.493）
- アイテムバンク生成時の分散設定が元論文と異なることが原因と考えられる

**EXP009・EXP010（data/3PL、500項目）**
- `src/generate_item_banks_3pl.R` で元論文に準拠した分散設定（a ~ N(1.2, 0.25), b ~ N(0,1), c ~ N(0.25, 0.02)）を使用したことで、MFI の RMSE が元論文の値に近づいた
- step 40 では元論文の 0.245 に対して 0.240〜0.245 とほぼ一致
- しかし **全ステップを通じて MFI の RMSE が元論文を上回る傾向**が残っており、特に序盤（step 10〜20）で差が大きい


