# EXP012: DQN with Correct Positive Constraint Order

## 目的

EXP007 との差は **正値制約（`Apply_Positive_Constraint`）の適用順** のみ。

| | EXP007 | EXP012 |
|---|---|---|
| 適用順 | `step()` の**前** → 制約が実質無効 | `step()` の**後** → 制約が正しく機能 |

## 背景

元論文（Wang et al., 2024, p.8703）の記述：

> "To ensure that the information calculated by the Q-network is positive, all network parameters are constrained to be positive."

この意図を実現するには、Adam が重みを更新した**後**に非負クランプを適用する必要がある。しかし元論文のオリジナルコードおよび EXP007 は `optimizer.step()` の**前**に制約を適用しており、直後の `step()` が重みを再び上書きするため制約が実質的に無効になっている。

## EXP007 からの変更点

変更箇所は TRAIN 関数内の2行の順序のみ：

```python
# EXP007（制約が無効）
optimizer.zero_grad()
loss.backward()
Apply_Positive_Constraint(eval_net)  # ← step() 前：直後の step() で上書きされる
optimizer.step()

# EXP012（制約が有効）
optimizer.zero_grad()
loss.backward()
optimizer.step()
Apply_Positive_Constraint(eval_net)  # ← step() 後：Adam の更新を非負にクランプ
```

他の設定（ネットワーク構造、バンク、γ、prior、報酬のθ、受検者数など）は EXP007 と完全に同一。

## シミュレーション設定

EXP007 と同一：

- アイテムバンク：`data/uncorrelated_banks/`（200項目）
- 受検者数：5,000 名、θ ~ N(0, 1)
- テスト長：40 問
- 特性値推定：MLE、範囲 [-4, 4]
- 初期特性値：Uniform(-0.5, 0.5)
- prior = "normal"
- γ = 0.1（予定）

## 出力

- `EXP012/models/dqn_normal_uncor_{bank_id}_gamma_{gamma}.pt` — 学習済みモデル
- `EXP012/results/summary_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — ステップ別 Bias・RMSE・MAE
- `EXP012/results/records_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — 全受検者の回答・推定値ログ

## 結果

（実験実施後に記入）

### EXP007 との比較（step 40 RMSE）

| γ | EXP007（制約無効） | EXP012（制約有効） |
|---:|---:|---:|
| 0.1 | — | — |

### 元論文との比較

| Test length | 元論文 DQN | 元論文 MFI | EXP007 DQN | EXP012 DQN |
|---:|---:|---:|---:|---:|
| 10 | 0.415 | 0.493 | — | — |
| 20 | 0.278 | 0.331 | — | — |
| 30 | 0.244 | 0.277 | — | — |
| 40 | 0.227 | 0.245 | — | — |

## 考察（実施前の仮説）

- 正値制約が機能することで Q ネットワークの出力（累積フィッシャー情報量の期待値）が常に非負に保たれ、学習が安定化する可能性がある
- 一方、制約が無効であっても ReLU 活性化関数と Kaiming 初期化により出力が自然に非負になりやすい構造のため、差が小さい可能性もある
- どちらが実際に優れるかは実験結果で判断する
