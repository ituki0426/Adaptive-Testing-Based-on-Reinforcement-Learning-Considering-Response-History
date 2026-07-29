# EXP016: DQN on 3PL Banks with Estimated θ̂ Reward（論文アルゴリズム準拠）

## 目的

EXP009 との差は **報酬の計算に使う θ の違い** のみ。

| | EXP009 | EXP016 |
|---|---|---|
| 報酬 | `FI(item_bank[action], training_theta[j])` — 真の θ（オラクル） | `FI(item_bank[action], state[-1:])` — 推定 θ̂（論文準拠） |

元論文 Wang et al. (2024) の Algorithm 1 では、報酬は推定特性値 θ̂_l で計算した Fisher 情報量 `I_{i_l}(θ̂_l)` と定義されている。EXP009（および EXP007〜EXP010）は真の θ で報酬を計算しており、論文の定式化と実装が乖離している。本実験は `data/3PL/` バンク上で論文準拠の報酬定義を検証する。

EXP011 は同じ報酬変更を `data/uncorrelated_banks/` で行ったものであり、EXP016 はそのバンクを `data/3PL/` に置き換えたバージョンに相当する。

## EXP009 からの変更点

変更箇所は TRAIN 関数内の1行のみ：

```python
# EXP009（真の θ を使用）
reward = FI(item_bank[action,], training_theta[j])

# EXP016（推定 θ̂ を使用）
reward = FI(item_bank[action,], state[-1:])
```

`state[-1]` は action 選択直前の推定特性値 θ̂_l。最初のステップでは Dodd 法の初期値（Uniform(-0.5, 0.5) から生成）が使われる。

他の設定（ネットワーク構造・バンク・γ・prior・受検者数・TD ターゲット計算）は EXP009 と完全に同一。

## 各実験との対応関係

| 変更内容 | EXP009 | EXP011 | EXP016 |
|---|---|---|---|
| data/3PL 使用 | ✓ | — | ✓ |
| 報酬を推定 θ̂ で計算 | — | ✓ | ✓ |

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

- `EXP016/models/dqn_normal_3pl_{bank_id}_gamma_{gamma}.pt` — 学習済みモデル
- `EXP016/results/records_3pl_{bank_id}_DQN_normal_gamma_{gamma}.csv` — 全受検者の回答・推定値ログ
- `EXP016/results/summary_3pl_{bank_id}_DQN_normal_gamma_{gamma}.csv` — ステップ別 Bias・RMSE・MAE
- `EXP016/results/rmse_comparison_3pl_{bank_id}_gamma_{gamma}.png` — MFI vs DQN プロット

MFI ベースラインは `data/3PL/results/summary_3pl_{bank_id}_MFI.csv` を参照。

## 結果

（実験実施後に記入）

### EXP009 との比較（step 40 RMSE）

| γ | EXP009（真の θ） | EXP016（推定 θ̂） |
|---:|---:|---:|
| 0.1 | 0.282 | — |

### 元論文との比較

| Test length | 元論文 DQN | 元論文 MFI | local MFI | EXP016 DQN |
|---:|---:|---:|---:|---:|
| 10 | 0.415 | 0.493 | 0.504 | — |
| 20 | 0.278 | 0.331 | 0.340 | — |
| 30 | 0.244 | 0.277 | 0.278 | — |
| 40 | 0.227 | 0.245 | 0.245 | — |

## 考察（実施前の仮説）

EXP011（`data/uncorrelated_banks/`）の結果では γ=0.1 の step 40 RMSE が EXP007 とほぼ同等（0.330 vs 0.329）だった。`data/3PL/` でも同様の傾向が期待されるが、バンクの分布が異なるため、序盤の推定誤差の影響が EXP011 と異なる可能性がある。
