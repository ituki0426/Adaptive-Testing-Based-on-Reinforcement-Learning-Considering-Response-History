# EXP010: DQN on 3PL Banks with RESPOND Bug Fix + TD Target Fix

## 目的

EXP007 に対して以下の3点を変更し、それ以外の学習アルゴリズムは EXP007 と同一に保つ：

1. **`RESPOND` の乱数生成バグ修正**（EXP008 由来）
2. **TD ターゲット計算の修正**（EXP008 由来）
3. **アイテムバンクを `data/3PL/` に変更**（EXP009 由来）

## EXP007 からの変更点

### 1. RESPOND バグ修正

```python
# EXP007（バグあり）
resp = (np.random.rand(1) <= p).astype(int)
# → スカラー乱数1個を全項目と比較するため、全項目が同じ正誤になる

# EXP010（修正後）
resp = (np.random.random(size=p.shape) <= p).astype(int)
# → 項目ごとに独立した乱数を生成
```

### 2. TD ターゲット計算の修正（最重要）

EXP007 の問題：replay buffer 経由の学習時に Q(next) を計算する際、その受検者がすでに出題した項目も選択肢として扱われる。

EXP010 の対策として、memory のフォーマットを拡張：

```
EXP007: (state, action, reward, next_state)                           → 幅 input_size*2+2
EXP010: (state, action, reward, next_state, terminal, next_available) → 幅 input_size*2+3+action_space
```

- `terminal`（float）：エピソード末尾フラグ
- `next_available`（size=action_space）：次ステップで未出題の項目マスク

TD ターゲットの計算：

```python
# EXP007
if i == test_length-1: q_target = batch_reward
else: q_target = batch_reward + gamma * q_next.max(1)[0]

# EXP010：出題済み項目を -inf でマスクし、終端状態では Q_next をゼロに
q_next     = q_next.masked_fill(~batch_next_mask, -torch.inf)
q_next_max = q_next.max(1)[0].view(cfg.batch_size, 1)
q_next_max = torch.where(batch_terminal.bool(), torch.zeros_like(q_next_max), q_next_max)
q_target   = batch_reward + cfg.gamma * q_next_max * (1.0 - batch_terminal)
```

### 3. アイテムバンクの変更

| 項目 | EXP007 | EXP010 |
|---|---|---|
| バンクディレクトリ | `data/uncorrelated_banks/` | `data/3PL/` |
| ファイル名 | `item_bank_{type}_{id}.csv` | `item_bank_uncor_{id}.csv` |
| 項目数（`n_items`） | 200 | 500 |
| `Config.bank_type` | あり | 削除（uncor 固定） |

## 各 EXP との変更の対応関係

| 変更内容 | EXP007 | EXP008 | EXP009 | EXP010 |
|---|---|---|---|---|
| RESPOND バグ修正 | — | ✓ | ✓ | ✓ |
| TD ターゲットのマスク修正 | — | ✓ | — | ✓ |
| data/3PL 使用 | — | — | ✓ | ✓ |
| MLE 数値安定化（clip） | — | ✓ | ✓ | — |

## シミュレーション設定

EXP007 と同一：

- 受検者数：5,000 名、θ ~ N(0, 1)
- テスト長：40 問
- γ = 0.1、prior = "normal"
- training_size = 1000、validation_interval = 50

## Colab でのデータパス

```
/content/drive/MyDrive/Colab Notebooks/Grad_Research/data/3PL/item_bank_uncor_{bank_id}.csv
/content/drive/MyDrive/Colab Notebooks/Grad_Research/data/theta_true/theta_true_{bank_id}.csv
```

## 出力

- `EXP010/models/dqn_normal_3pl_{bank_id}_gamma_0.1.pt`
- `EXP010/results/records_3pl_{bank_id}_DQN_normal_gamma_0.1.csv`
- `EXP010/results/summary_3pl_{bank_id}_DQN_normal_gamma_0.1.csv`

## MFI ベースライン（参考）

`src/Test MFI on the 3PL bank.R` で bank_id=1 を実行した結果：

| Bias | RMSE | MAE |
|---|---|---|
| 0.008 | 0.245 | 0.191 |
