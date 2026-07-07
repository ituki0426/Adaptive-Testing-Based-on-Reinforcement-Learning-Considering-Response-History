# EXP008: DQN with TD Target Fix and Available Item Masking

## 目的

EXP007 の DQN 実装に存在する、TD ターゲット計算上のバグを修正する。具体的には、replay buffer からサンプルした遷移で Q(next) を計算する際に**出題済み項目が次の行動候補に含まれてしまう問題**を解消し、正しい項目選択制約のもとで学習させる。

## EXP007 からの変更点

### 1. バグ修正

| 箇所 | EXP007 | EXP008 |
|---|---|---|
| `RESPOND` の乱数生成 | `np.random.rand(1)` — スカラー1個のみ生成 | `np.random.random(size=p.shape)` — 項目ごとに独立した乱数 |
| `MLE` / `MLE_TEST` の数値安定化 | なし | `p = np.clip(p, 1e-10, 1 - 1e-10)` を追加（`log(0)` 防止） |

### 2. TD ターゲット計算の修正（最重要）

EXP007 の問題：replay buffer 経由の学習時に Q(next) を計算する際、その受検者がすでに出題した項目も選択肢として扱われる。

EXP008 の対策として、memory のフォーマットを拡張：

```
EXP007: (state, action, reward, next_state)                          → 幅 input_size*2+2
EXP008: (state, action, reward, next_state, terminal, next_available) → 幅 input_size*2+3+action_space
```

- `terminal`（float）：エピソード末尾フラグ
- `next_available`（size=action_space）：次ステップで未出題の項目マスク

TD ターゲットの計算：

```python
# EXP007
if i == test_length-1: q_target = batch_reward
else: q_target = batch_reward + gamma * q_next.max(1)[0]

# EXP008：出題済み項目を -inf でマスクし、終端状態では Q_next をゼロに
q_next = q_next.masked_fill(~batch_next_mask, -torch.inf)
q_next_max = torch.where(batch_terminal.bool(), zeros, q_next.max(1)[0])
q_target = batch_reward + cfg.gamma * q_next_max * (1.0 - batch_terminal)
```

### 3. ベストモデルの保存方法

| 項目 | EXP007 | EXP008 |
|---|---|---|
| 保存タイミング | バリデーション中にその場でファイル保存 | `copy.deepcopy(state_dict)` でメモリ保持し、訓練終了後に保存 |
| 保存形式 | `torch.save(model, ...)` — モデルごと（`.t7`） | `torch.save(state_dict, ...)` — state_dict のみ（`.pt`） |
| モデル選択基準 | Bias・RMSE・MAE の3指標すべて改善 | RMSE のみで比較 |

### 4. エンジニアリング改善

- **`Config` dataclass** の導入：グローバル変数だったハイパーパラメータを一元管理
- **`find_project_root()`**：Windows ドライブパス（`E:/...`）のハードコードをなくし、Colab・ローカル双方で動作するポータブルなパス解決
- **結果 CSV の追加**：`records_*.csv` に加え、ステップ別集計の `summary_*.csv` も保存

## シミュレーション設定

EXP007 と同一：

- アイテムバンク：`data/uncorrelated_banks/`、500 項目
- 受検者数：5,000 名、θ ~ N(0, 1)
- テスト長：40 問
- γ = 0.1、prior = "normal"

## 出力

- `EXP007/models/dqn_normal_uncor_{bank_id}_gamma_0.1.pt` — 学習済みモデル（EXP008 は EXP007 の model_dir を共用）
- `EXP007/results/records_uncor_{bank_id}_DQN_normal_gamma_0.1.csv`
- `EXP007/results/summary_uncor_{bank_id}_DQN_normal_gamma_0.1.csv`

## 仮説

出題済み項目をマスクした正しい TD ターゲットで学習することで、EXP007 より低い RMSE が得られる。
