# EXP011: DQN with Estimated θ̂ Reward（論文アルゴリズム準拠）

## 目的

EXP007 との差は **報酬の計算に使う θ の違い** のみ。

| | EXP007 | EXP011 |
|---|---|---|
| 報酬 | I_{i_l}(θ\_true) — 真の θ（オラクル） | I_{i_l}(θ̂\_l) — 推定値（論文準拠） |

元論文 Wang et al. (2024) のアルゴリズム（Algorithm 1）では、報酬は推定特性値 θ̂\_l で計算した Fisher 情報量 I_{i_l}(θ̂\_l) と定義されている。ところが元論文のオリジナルコード（および EXP007〜EXP010）は真の θ で報酬を計算しており、論文の定式化と実装が乖離している。

本実験は論文のアルゴリズム定義に忠実な実装で DQN を訓練し、その影響を EXP007 と比較することで、この乖離が性能に与える効果を定量的に評価する。

## 論文の記述（Wang et al., 2024）

Algorithm 1 ステップ3・4（p.8700）：

> (3) Calculate the Fisher information **I\_{i\_l}(θ̂\_l)** and obtain the updated trait level θ̂\_{l+1} by the estimation method.
> (4) Store **{ θ̂\_l, i\_l, I\_{i\_l}(θ̂\_l), θ̂\_{l+1} }** in D.

損失関数（Eq. 13）：
$$\mathcal{L}(\Theta) = \left[I_{i_l}(\hat{\theta}_l) - Q_{\mathrm{action}}(\hat{\theta}_l, i_l; \Theta)\right]^2 \quad (l = L)$$
$$\mathcal{L}(\Theta) = \left[I_{i_l}(\hat{\theta}_l) + \gamma \max_{i} Q_{\mathrm{target}}(\hat{\theta}_{l+1}, i; \Theta') - Q_{\mathrm{action}}(\hat{\theta}_l, i_l; \Theta)\right]^2 \quad (\text{otherwise})$$

## EXP007 からの変更点

変更箇所は TRAIN 関数内の1行のみ：

```python
# EXP007（真の θ を使用）
reward = FI(item_bank[action,], training_theta[j])

# EXP011（推定 θ̂ を使用）
reward = FI(item_bank[action,], state[-1:])
```

`state[-1]` は現在の推定特性値 θ̂\_l。最初のステップでは Dodd 法の初期値（Uniform(-0.5, 0.5) から生成）が使われる。

他の設定（ネットワーク構造、バンク、γ、prior、受検者数など）は EXP007 と完全に同一。

## シミュレーション設定

EXP007 と同一：

- アイテムバンク：`data/uncorrelated_banks/`（500項目）
- 受検者数：5,000 名、θ ~ N(0, 1)
- テスト長：40 問
- 特性値推定：MLE、範囲 [-4, 4]
- 初期特性値：Uniform(-0.5, 0.5)
- prior = "normal"
- 実施した γ：0.1（予定）

## 出力

- `EXP011/models/dqn_normal_uncor_{bank_id}_gamma_{gamma}.pt` — 学習済みモデル
- `EXP011/results/summary_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — ステップ別 Bias・RMSE・MAE
- `EXP011/results/records_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — 全受検者の回答・推定値ログ

## 結果

（実験実施後に記入）

### EXP007 との比較（step 40 RMSE）

| γ | EXP007（真の θ） | EXP011（推定 θ̂） |
|---:|---:|---:|
| 0.1 | — | — |

### 元論文との比較

| Test length | 元論文 DQN | 元論文 MFI | EXP007 MFI | EXP011 DQN |
|---:|---:|---:|---:|---:|
| 10 | 0.415 | 0.493 | 0.701 | — |
| 20 | 0.278 | 0.331 | 0.462 | — |
| 30 | 0.244 | 0.277 | 0.377 | — |
| 40 | 0.227 | 0.245 | 0.336 | — |

## 考察（実施前の仮説）

- 推定 θ̂ は序盤（step 1〜5）ほど真の θ との乖離が大きいため、序盤の報酬信号のノイズが増加する可能性がある
- 結果として、序盤の訓練が不安定になり RMSE が EXP007 より悪化する可能性がある
- 一方、テスト時の Q 値計算も θ̂ を入力とするため、訓練と評価の θ の分布が一致し、EXP007 より良化する可能性もある
- どちらが実際に優れるかは実験結果で判断する
