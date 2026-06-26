# EXP004: DQN の状態を「θ̂ + 事後分散」の2次元に拡張する実装メモ

## 1. 目的

ベースライン（EXP003）の DQN は、状態を現在の推定特性値 θ̂（スカラー, `input_size = 1`）のみとしている。
しかし CLAUDE.md でも指摘している通り、**同じ θ̂ でも蓄積された情報量（推定の確からしさ）が異なる**ため、
スカラーの θ̂ だけでは不確実性の違いを区別できない。

本実験（EXP004）では、DQN ネットワークへの入力を

$$ s_t = (\hat\theta_t,\; \mathrm{Var}(\hat\theta_t)) $$

の **2次元** に拡張し、推定値そのものに加えて**事後分散（推定の不確実性）**を状態として与える。

## 2. 設計方針（確定事項）

| 項目 | 方針 |
|---|---|
| 状態の次元 | 2次元固定 `[θ̂, var]`（`input_size = 2`、履歴入力 `input_size>1` の汎用機能は使わない） |
| θ̂ の推定法 | 従来どおり **MLE**（変更なし） |
| 事後分散の計算 | **MLE の漸近分散で近似**: `var ≈ 1 / I(θ̂)`、`I(θ̂) = Σ_k FI(item_k, θ̂)`（出題済み項目のフィッシャー情報量の和） |
| 報酬 | 変更なし（学習時は真の θ でのフィッシャー情報量） |
| 評価指標 | 変更なし（Bias / RMSE / MAE） |

> 注: 「事後分散」という語を用いるが、ここでは厳密なベイズ事後分布の分散ではなく、
> MLE 推定量の漸近分散（テスト情報量の逆数 = 標準誤差の二乗）で近似する。
> 出題済み項目数が増えるほど `I(θ̂)` が大きくなり、`var` が小さくなる挙動をとる。

## 3. 状態の定義と端点処理

### 3.1 状態ベクトルの並び

```
state = [θ̂, var]      # state[0] = θ̂, state[1] = var
```

> ⚠️ 既存コードは「θ̂ を末尾要素 `state[-1]`」として参照している箇所が多い
> （履歴ゼロ詰めが先頭に来る設計のため）。本実装では並びを `[θ̂, var]` に固定し、
> **θ̂ を参照する箇所はすべて `state[-1]` → `state[0]`（ベクトル化版は `state[-1,:]` → `state[0,:]`）に書き換える**。

### 3.2 事後分散の計算式

出題済み項目集合を $\{i_1,\dots,i_t\}$、現在の推定値を $\hat\theta$ とすると

$$ I(\hat\theta) = \sum_{k=1}^{t} I_{i_k}(\hat\theta),\qquad \mathrm{Var}(\hat\theta) \approx \frac{1}{I(\hat\theta)} $$

### 3.3 端点処理（重要）

| 状況 | 扱い |
|---|---|
| 出題 0 問（初期状態） | `I = 0` で 1/I が発散するため、**初期分散を事前分散 = 1.0 とする**（θ̂ ~ N(0,1) の事前を想定） |
| 全問正答 / 全問誤答（MLE が境界で不定） | θ̂ は既存のヒューリスティック（中点法）で決め、その θ̂ における `I(θ̂)` から `var = 1/I` を計算（項目が1問以上あれば `I>0`） |
| `I` が小さく `1/I` が大きすぎる | 数値安定のため上限 `max_var = 1.0`（事前分散）でクリップ。事後の不確実性が事前を超えない、という解釈とも整合 |

`max_var` はパラメータ化し、後でクリップ有無・値を実験で比較できるようにしておく。

## 4. 具体的なコード変更箇所

対象ファイル: `EXP004/src/Train and test DQN on the simulated banks.py`
（実データ版 `Train and test DQN on the real responses.py` も同様の変更が必要）

### 4.1 ハイパーパラメータ

```python
# 変更前
input_size = 1
# 変更後
input_size = 2          # [θ̂, var]
prior_var = 1.0         # 初期分散 = 事前分散
max_var   = 1.0         # 事後分散のクリップ上限
```

### 4.2 事後分散を計算するヘルパー関数を追加

`FI` の直後あたりに追加する。

```python
### 出題済み項目のテスト情報量から MLE 漸近分散（≒事後分散）を計算（学習フェーズ用・スカラー）
def POSTERIOR_VAR(item_paras, theta, max_var=1.0):
    info = FI(item_paras, theta).sum()          # I(θ̂) = Σ_k FI_k(θ̂)
    if info <= 0:
        return max_var
    return float(min(1.0 / info, max_var))

### 検証・テストフェーズ用（受検者方向にベクトル化）
### item_paras: shape (n_administered, N, 3), theta: shape (N,)  ->  return: shape (N,)
def POSTERIOR_VAR_TEST(item_paras, theta, max_var=1.0):
    var = np.zeros(theta.shape[0])
    for j in range(theta.shape[0]):
        info = FI(item_paras[:, j, :], theta[j]).sum()
        var[j] = max_var if info <= 0 else min(1.0 / info, max_var)
    return var
```

> `FI(item_para, theta)` は `item_para[:,0/1/2]` で a,b,c を取り出す実装のため、
> `item_bank[item_id]`（1D index → shape `(n,3)`）や `item_paras[:,j,:]`（shape `(n,3)`）をそのまま渡せる。

### 4.3 TRAIN：初期状態

```python
# 変更前
state = np.concatenate((np.zeros(input_size-1), np.random.rand(1) - 0.5))
# 変更後（state = [θ̂, var]）
theta_hat = np.random.rand(1) - 0.5         # 初期 θ̂ ~ U(-0.5, 0.5)
state = np.concatenate((theta_hat, [prior_var]))   # [θ̂, 初期分散]
```

### 4.4 TRAIN：next_state の構築（ループ内）

```python
# 変更前
if len(np.unique(resp)) == 1 :
    if resp[-1] == 1 :
        next_state = np.array([state[-1] + (item_bank[:,1].max() - state[-1]) / 2])
    else :
        next_state = np.array([state[-1] - (state[-1] - item_bank[:,1].min()) / 2])
else : next_state = MLE(item_bank[item_id,], resp)
if input_size > 1 : next_state = np.concatenate((state[-(input_size-1):], next_state))

# 変更後
theta_prev = state[0]                                  # ← state[-1] ではなく state[0]
if len(np.unique(resp)) == 1 :
    if resp[-1] == 1 :
        next_theta = np.array([theta_prev + (item_bank[:,1].max() - theta_prev) / 2])
    else :
        next_theta = np.array([theta_prev - (theta_prev - item_bank[:,1].min()) / 2])
else :
    next_theta = MLE(item_bank[item_id,], resp)
next_var   = POSTERIOR_VAR(item_bank[item_id], next_theta[0], max_var)
next_state = np.concatenate((next_theta, [next_var]))  # [θ̂', var']
```

> **memory / バッチ取り出しは変更不要**。`memory = np.zeros((memory_capacity, input_size*2+2))`
> および `batch_state = memory[:, :input_size]` / `batch_next_state = memory[:, -input_size:]` 等は
> すべて `input_size` でスライスしているため、`input_size = 2`（6 列: state2 + action1 + reward1 + next_state2）に自動対応する。
> `Choose_Action` も `state` ベクトルをそのまま投入するため変更不要。

### 4.5 検証（Validation, TRAIN 内）

初期状態（行 = 状態次元, 列 = 受検者）:

```python
# 変更前
state = np.concatenate((np.zeros((input_size-1, validation_size)),
                        np.expand_dims((np.random.rand(validation_size)-0.5), axis=0)))
# 変更後（行0 = θ̂, 行1 = var）
theta_hat0 = (np.random.rand(validation_size) - 0.5)[np.newaxis, :]
var0       = np.full((1, validation_size), prior_var)
state      = np.vstack((theta_hat0, var0))
```

ヒューリスティック中の θ̂ 参照を `state[-1,...]` → `state[0,...]` に変更:

```python
# 変更前
theta_0[idx_full] = state[-1,idx_full] + (item_bank[:,1].max() - state[-1,idx_full]) / 2
theta_0[idx_zero] = state[-1,idx_zero] + (item_bank[:,1].min() - state[-1,idx_zero]) / 2
# 変更後
theta_0[idx_full] = state[0,idx_full] + (item_bank[:,1].max() - state[0,idx_full]) / 2
theta_0[idx_zero] = state[0,idx_zero] + (item_bank[:,1].min() - state[0,idx_zero]) / 2
```

状態更新（次状態の作成）を分散込みに変更:

```python
# 変更前
if input_size > 1 : state = np.concatenate((state[-(input_size-1):], theta_0[np.newaxis, :]))
else: state = theta_0[np.newaxis, :]
# 変更後（行0 = θ̂', 行1 = var'）
var_0 = POSTERIOR_VAR_TEST(item_bank[item_id], theta_0, max_var)   # item_id は今ステップの項目を含む
state = np.vstack((theta_0[np.newaxis, :], var_0[np.newaxis, :]))
```

> `Choose_Action_Test` 内の `state.swapaxes(0,1)`（→ shape `(N, input_size)`）はそのまま機能する。

### 4.6 TEST

初期状態:

```python
# 変更前
state = np.concatenate((np.zeros((input_size-1, testing_size)),
                        np.expand_dims((np.random.rand(testing_size)-0.5), axis=0)))
# 変更後
theta_hat0 = (np.random.rand(testing_size) - 0.5)[np.newaxis, :]
var0       = np.full((1, testing_size), prior_var)
state      = np.vstack((theta_hat0, var0))
```

ヒューリスティックの θ̂ 参照を `state[-1,...]` → `state[0,...]`:

```python
theta_0[:,idx_full] = state[0,idx_full] + (item_bank[:,1].max() - state[0,idx_full]) / 2
theta_0[:,idx_zero] = state[0,idx_zero] + (item_bank[:,1].min() - state[0,idx_zero]) / 2
```

状態更新を分散込みに変更（`theta_0` は shape `(1, N)`）:

```python
# 変更前
if input_size > 1 : state = np.concatenate((state[-(input_size-1):], theta_0))
else: state = theta_0
# 変更後
var_0 = POSTERIOR_VAR_TEST(item_bank[item_id], theta_0[0], max_var)   # shape (N,)
state = np.vstack((theta_0, var_0[np.newaxis, :]))
```

### 4.7 出力ファイル名（ベースラインと混同しないため）

ベースライン DQN の結果を上書きしないよう、保存名にサフィックスを付ける。

```python
# モデル
model_path = MODEL_DIR / f"dqn_var_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.t7"
# 結果 CSV（TEST 内）
dqn_data.to_csv(RESULTS_DIR / f"records_{bank_type}_{bank_id}_DQNvar_{prior}_gamma_{gamma}.csv", index=False)
```

## 5. 注意点・検討事項

1. **入力スケールの違い**: θ̂ は概ね `[-4, 4]`、var は `[0, 1]` とスケールが異なる。
   既存コードは `Apply_Positive_Constraint` で全パラメータを非負にクランプしている。
   まずはベースラインと同条件で動かし、必要なら var を標準化（例: `var * c`）してスケールを揃えることを検討。
2. **クリップの是非**: `max_var = 1.0` で頭打ちにすると、序盤（1〜数問）の「分散が大きい」情報が潰れる可能性がある。
   `max_var` を緩める（例 5.0）/ クリップなし、の比較も実験候補。
3. **初期分散の値**: `prior_var = 1.0` は θ ~ N(0,1) の事前分散に対応。妥当だが感度確認の余地あり。
4. **公平な比較**: バンク・乱数シード・テスト長・γ・prior 等はベースライン（EXP003 / EXP004 既存 DQN）と揃える。

## 6. 動作確認手順

1. `input_size = 2` に変更し、まず小さい `training_size`（例 50）でスタックトレースなく1周回ることを確認。
2. 検証ログ（`step_valid`）の Bias/RMSE/MAE が NaN を含まないこと、var が `[0, max_var]` に収まることを確認。
3. 本番設定で学習 → `TEST` → `records_..._DQNvar_....csv` を出力。
4. ベースライン DQN（1次元状態）と同一バンクで RMSE/MAE/Bias をステップ別に比較し、
   特に**序盤での改善有無**を確認する。
