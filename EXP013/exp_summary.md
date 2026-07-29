# EXP013: DQN with Correct Target Network Initialization（論文準拠）

## 目的

EXP007 との差は **target network の初期化方法** のみ。

| | EXP007 | EXP013 |
|---|---|---|
| target network 初期化 | `target_net.initialize()` — eval_net と独立した Kaiming 初期化（Θ′ ≠ Θ） | `target_net.load_state_dict(eval_net.state_dict())` — eval_net のコピー（Θ′ = Θ） |

## 背景

元論文（Wang et al., 2024, p.8700）の記述：

> "Initialize the action-value network Q_action with random weights **Θ**, the target network Q_target with random weights **Θ′ = Θ**"

論文は target network を eval network と**同じ重みで初期化**することを明示している。しかし EXP007 は `eval_net.initialize()` と `target_net.initialize()` を独立に呼び出しており、両ネットワークが異なる乱数で初期化される（Θ′ ≠ Θ）。

## EXP007 からの変更点

変更箇所は cell-8 の1行のみ：

```python
# EXP007（論文と不一致）
eval_net.initialize()
target_net.initialize()       # 独立した Kaiming 初期化 → Θ′ ≠ Θ

# EXP013（論文準拠）
eval_net.initialize()
target_net.load_state_dict(eval_net.state_dict())  # Θ′ = Θ
```

他の設定（ネットワーク構造、バンク、γ、prior、報酬の θ、受検者数など）は EXP007 と完全に同一。

## シミュレーション設定

EXP007 と同一：

- アイテムバンク：`data/uncorrelated_banks/`（500項目）
- 受検者数：5,000 名、θ ~ N(0, 1)
- テスト長：40 問
- 特性値推定：MLE、範囲 [-4, 4]
- 初期特性値：Uniform(-0.5, 0.5)
- prior = "normal"
- γ = 0.1（予定）

## 出力

- `EXP013/models/dqn_normal_uncor_{bank_id}_gamma_{gamma}.pt` — 学習済みモデル
- `EXP013/results/summary_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — ステップ別 Bias・RMSE・MAE
- `EXP013/results/records_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — 全受検者の回答・推定値ログ

## 結果

（実験実施後に記入）

### EXP007 との比較（step 40 RMSE）

| γ | EXP007（Θ′ ≠ Θ） | EXP013（Θ′ = Θ） |
|---:|---:|---:|
| 0.1 | — | — |

### 元論文との比較

| Test length | 元論文 DQN | 元論文 MFI | EXP007 MFI | EXP013 DQN |
|---:|---:|---:|---:|---:|
| 10 | 0.415 | 0.493 | 0.701 | — |
| 20 | 0.278 | 0.331 | 0.462 | — |
| 30 | 0.244 | 0.277 | 0.377 | — |
| 40 | 0.227 | 0.245 | 0.336 | — |

## 考察（実施前の仮説）

- 訓練開始時に eval_net と target_net が同一重みを持つことで、初期の TD ターゲットが eval_net 自身の出力と一致し、訓練初期の安定性が変化する可能性がある
- 学習が進むにつれ、周期的な `load_state_dict` によって両ネットワークは同期されるため、長期的な影響は収束後の差として現れる
- EXP007 との差が小さければ、初期化のずれは学習によって自然に吸収されると解釈できる
