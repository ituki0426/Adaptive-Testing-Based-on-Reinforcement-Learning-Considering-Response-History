# EXP014: DQN with Train/Eval Loss・RMSE Logging

## 目的

EXP013 に**学習ログ（Train Loss・Eval Loss・Train RMSE・Eval RMSE）の記録と出力**を追加した実験。アルゴリズム本体は EXP013 と完全に同一。

## EXP013 からの変更点

TRAIN 関数内にログ収集ロジックを追加したのみ。

### 1. Train Loss

各勾配更新ステップで `loss.item()` を記録し、バリデーション間隔（50エピソード）ごとに平均を計算する。

```python
interval_losses.append(loss.item())
# バリデーション時に集計
train_loss = float(np.mean(interval_losses))
```

### 2. Eval Loss

バリデーション時に、リプレイバッファから新規サンプル（batch_size=128）を引いて TD 誤差を計算する。重み更新は行わない。

```python
with torch.no_grad():
    qe = eval_net(bs).gather(1, ba)
    qn = target_net(bns)
    qt = br + cfg.gamma * qn.max(1)[0].view(cfg.batch_size, 1)
    eval_loss = float(loss_func(qe, qt).item())
```

### 3. Train RMSE（step 7-40 平均）

各訓練エピソードの step 7〜40 において、`next_state[-1]`（その時点の θ̂）と真の θ の二乗誤差を記録し、バリデーション間隔ごとに RMSE を計算する。

```python
if i >= 6:  # step 7 以降（0-indexed で i=6）
    interval_sq_errors.append(float((next_state[-1] - training_theta[j]) ** 2))
# バリデーション時に集計
train_rmse = float(np.sqrt(np.mean(interval_sq_errors)))
```

### 4. Eval RMSE（step 7-40 平均）

バリデーションシミュレーションで得た `step_valid` から、step 7-40 の RMSE を平均する。

```python
eval_rmse = float(np.mean(step_valid[6:, 2]))
```

### 5. TRAIN の戻り値の変更

```python
# EXP013
return best_state

# EXP014
return best_state, pd.DataFrame(train_log)
```

### 6. 出力ファイルの追加

```python
train_log.to_csv(RESULTS_DIR / f"train_log_{stem}.csv", index=False)
```

## シミュレーション設定

EXP013 と同一：

- アイテムバンク：`data/uncorrelated_banks/`（500項目）
- 受検者数：5,000 名、θ ~ N(0, 1)
- テスト長：40 問
- 特性値推定：MLE、範囲 [-4, 4]
- 初期特性値：Uniform(-0.5, 0.5)
- prior = "normal"
- γ = 0.1（予定）

## 出力

- `EXP014/models/dqn_normal_uncor_{bank_id}_gamma_{gamma}.pt` — 学習済みモデル
- `EXP014/results/records_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — 全受検者の回答・推定値ログ
- `EXP014/results/summary_uncor_{bank_id}_DQN_normal_gamma_{gamma}.csv` — ステップ別 Bias・RMSE・MAE
- `EXP014/results/train_log_normal_uncor_{bank_id}_gamma_{gamma}.csv` — バリデーションごとの Loss・RMSE ログ

## ノートブック

- `notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb` — 学習・テスト本体
- `notebook/plot_train_log.ipynb` — Train/Eval Loss・RMSE の学習曲線プロット
- `notebook/plot_rmse_comparison.ipynb` — MFI vs DQN の RMSE 比較プロット

## 結果

（実験実施後に記入）

### 学習曲線

（`plot_train_log.ipynb` 実行後に記入）

### EXP013 との比較（step 40 RMSE）

| γ | EXP013 | EXP014 |
|---:|---:|---:|
| 0.1 | — | — |
