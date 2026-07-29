# EXP015: DQN on 3PL Banks with TD Target Fix

## 目的

EXP009（`data/3PL/`、TD ターゲットマスクなし）に、EXP008 で導入した**TD ターゲット計算の修正**を適用した実験。

各実験との対応関係：

| 変更内容 | EXP009 | EXP010 | EXP015 |
|---|---|---|---|
| data/3PL 使用 | ✓ | ✓ | ✓ |
| TD ターゲットのマスク修正 | — | ✓ | ✓ |
| RESPOND バグ修正 | ✓ | ✓ | ✓ |
| MLE 数値安定化（clip） | ✓ | — | ✓ |

EXP010 との違いは MLE 数値安定化（`np.clip`）の有無のみ。

## EXP009 からの変更点

### メモリフォーマットの拡張

```
EXP009: (state, action, reward, next_state)                           → 幅 input_size*2+2
EXP015: (state, action, reward, next_state, terminal, next_available) → 幅 input_size*2+3+action_space
```

- `terminal`（float）：エピソード末尾フラグ（`i == test_length - 1` のとき 1.0）
- `next_available`（size=action_space）：次ステップで未出題の項目マスク

### TD ターゲット計算

```python
# EXP009
q_target = batch_reward if i == cfg.test_length - 1 \
           else batch_reward + cfg.gamma * q_next.max(1)[0].view(cfg.batch_size, 1)

# EXP015：出題済み項目を -inf でマスクし、終端状態では Q_next をゼロに
q_next     = q_next.masked_fill(~batch_next_mask, -torch.inf)
q_next_max = q_next.max(1)[0].view(cfg.batch_size, 1)
q_next_max = torch.where(batch_terminal.bool(), torch.zeros_like(q_next_max), q_next_max)
q_target   = batch_reward + cfg.gamma * q_next_max * (1.0 - batch_terminal)
```

## シミュレーション設定

EXP009 と同一：

- アイテムバンク：`data/3PL/`（500項目）
- 受検者数：5,000 名、θ ~ N(0, 1)
- テスト長：40 問
- 特性値推定：MLE、範囲 [-4, 4]
- 初期特性値：Uniform(-0.5, 0.5)
- prior = "normal"
- γ = 0.1（予定）

## 出力

- `EXP015/models/dqn_normal_3pl_{bank_id}_gamma_{gamma}.pt` — 学習済みモデル
- `EXP015/results/records_3pl_{bank_id}_DQN_normal_gamma_{gamma}.csv` — 全受検者の回答・推定値ログ
- `EXP015/results/summary_3pl_{bank_id}_DQN_normal_gamma_{gamma}.csv` — ステップ別 Bias・RMSE・MAE
- `EXP015/results/rmse_comparison_3pl_{bank_id}_gamma_{gamma}.png` — MFI vs DQN プロット

MFI ベースラインは `data/3PL/results/summary_3pl_{bank_id}_MFI.csv` を参照。

## 結果

（実験実施後に記入）

### EXP009 との比較（step 40 RMSE）

| γ | EXP009（マスクなし） | EXP015（マスクあり） |
|---:|---:|---:|
| 0.1 | 0.282 | — |

### 元論文との比較

| Test length | 元論文 DQN | 元論文 MFI | local MFI | EXP015 DQN |
|---:|---:|---:|---:|---:|
| 10 | 0.415 | 0.493 | 0.504 | — |
| 20 | 0.278 | 0.331 | 0.340 | — |
| 30 | 0.244 | 0.277 | 0.278 | — |
| 40 | 0.227 | 0.245 | 0.245 | — |
