# EXP007: DQN Baseline on Uncorrelated Banks

## 目的

Wang et al. (2024) の DQN ベース項目選択法を、元論文コード（`Train and test DQN on the simulated banks.py`）からローカル環境で再現できるよう整備した実験。元コードのパスハードコード・バグを修正しつつ、アルゴリズム本体は元論文と同一に保つことで、以降の実験（EXP008〜）のベースラインとする。

## 元論文コード（ルートの `Train and test DQN on the simulated banks.py`）からの変更点

### 1. エンジニアリング改善

| 項目 | 元コード | EXP007 |
|---|---|---|
| パス管理 | `E:/...` の Windows パスハードコード | `pathlib.Path` + `find_project_root()` でポータブルに解決 |
| ハイパーパラメータ管理 | グローバル変数 | `@dataclass Config` に一元化 |
| `action_space` | `500` ハードコード | `item_bank.shape[0]`（バンクサイズから自動取得） |
| 結果保存 | records CSV のみ | records CSV + ステップ別 summary CSV |
| モデル保存形式 | `torch.save(model, ...)` — モデルごと | `torch.save(state_dict, ...)` — state_dict のみ（`.pt`） |

### 2. バグ修正

| 箇所 | 元コード | EXP007 |
|---|---|---|
| `RESPOND` の乱数生成 | `np.random.rand(1)` — スカラー1個のみ | `np.random.random(size=p.shape)` — 項目ごとに独立した乱数 |
| `Choose_Action` の乱数 | `np.random.randn()` — 標準正規分布 | `np.random.rand()` — 一様分布（ε-greedy の判定として正しい） |

### 3. ベストモデルの保存方法

| 項目 | 元コード | EXP007 |
|---|---|---|
| 保存タイミング | バリデーション中にその場でファイル保存 | `copy.deepcopy(state_dict)` でメモリ保持し、訓練終了後に保存 |
| モデル選択基準 | Bias・RMSE・MAE の3指標すべて改善 | RMSE のみで比較 |

## アルゴリズム（元論文と同一）

- **状態**：推定特性値 θ̂（スカラー、MDP 定式化）
- **行動**：未出題項目の選択
- **報酬**：フィッシャー情報量 $I_i(\theta)$（真の θ で計算）
- **TD ターゲット**：`batch_reward + γ * Q_next.max()`（終端判定は step インデックスで行う）
- **ネットワーク**：全結合 3 層（50 → 30 → action_space）、Dropout なし

## シミュレーション設定

| 項目 | 値 |
|---|---|
| アイテムバンク | `data/uncorrelated_banks/item_bank_uncor_{id}.csv`（n_items=500、実効項目数はバンクサイズに依存） |
| 受検者数 | 5,000 名、θ ~ N(0, 1) |
| テスト長 | 40 問 |
| γ | 0.1 |
| prior | normal |
| training_size | 1,000 |
| validation_interval | 50 |

## 出力

- `EXP007/models/dqn_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.pt`
- `EXP007/results/records_{bank_type}_{bank_id}_DQN_{prior}_gamma_{gamma}.csv`
- `EXP007/results/summary_{bank_type}_{bank_id}_DQN_{prior}_gamma_{gamma}.csv`

## 以降の実験との関係

| 実験 | EXP007 からの主な変更 |
|---|---|
| EXP008 | TD ターゲットに出題済み項目マスクと terminal フラグを追加 |
| EXP009 | バンクを `data/3PL/`（500 項目）に変更 |
| EXP010 | EXP007 + RESPOND 修正 + TD マスク修正 + `data/3PL/` |
