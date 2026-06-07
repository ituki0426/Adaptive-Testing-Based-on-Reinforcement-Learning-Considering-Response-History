# EXP002 設計書

## 実験名
Burn-in DRQN

## 目的
DRQN の初期 RMSE 悪化に対して、replay memory から取り出した系列をそのまま全長で誤差逆伝播する代わりに、先頭数ステップを burn-in 区間として hidden state の構築に専念させる学習法の効果を確認する。

## 背景
- DRQN では系列の冒頭で hidden state が未成熟なまま TD 誤差を受ける。
- 序盤の不安定な hidden state に対して直接勾配を当てると、履歴表現の形成が不安定になる場合がある。
- burn-in は「先頭の数ステップは状態復元だけに使い、その後半だけで損失を計算する」定番の対処法である。

## ベースライン
- ベースコードは `EXP003/notebook/Train and test DRQN on the simulated banks.ipynb` の simulated-bank DRQN を基準にする。
- モデル構造は通常 DRQN のままとし、変更点を学習ループに限定する。
- 報酬は真の `theta` で計算した Fisher 情報量。

## 実装方針
- 実験コードは `EXP002/src/Train and test DRQN on the simulated banks.py` に置く。
- replay memory には従来どおりエピソード全体を保存する。
- 学習時は `burn_in_steps` だけを no-grad で LSTM に流し、そこで得た hidden state を suffix 学習の初期状態として使う。
- 損失は先頭 `burn_in_steps` ステップ（1-indexed の問題番号で 1〜`burn_in_steps` 問目）を除外し、残り（step `burn_in_steps + 1`〜`test_length`）の区間のみで計算する。`burn_in_steps = 5` なら step 6〜40 が学習対象。コード上は `resps_t[:, burn_in_steps:]` の slice に対応する（0-indexed 位置では `pos >= burn_in_steps`）。
- target Q の item mask は全履歴に基づいて計算し、burn-in より前に選ばれた項目も再選択されないようにする。
- validation / test の推論方策自体はベースラインから変えない。

## 期待する効果
- hidden state がある程度安定してから TD 学習が始まるため、初期数問の不安定さを緩和できる可能性がある。
- とくに replay された系列の先頭側で起きる noisy な更新を減らすことを狙う。

## 主要ハイパーパラメータ
- `test_length = 40`
- `embed_dim = 16`
- `lstm_hidden = 64`
- `gamma = 0.1`
- `memory_capacity = 1000`
- `batch_size = 128`
- `burn_in_steps = 5`
- `training_size = 1000`
- `validation_size = 200`

## 出力
- モデル: `EXP002/models/`
- 受検者ごとの詳細記録: `EXP002/results/records_*.csv`
- ステップ別要約: `EXP002/results/summary_*.csv`

## 比較観点
- 通常 DRQN と比べて `step 1-10` の RMSE が改善するか確認する。
- `burn_in_steps` を増やしすぎると学習に使う時点数が減るため、`step 40` の性能低下にも注意する。
- 序盤改善が弱い場合は `burn_in_steps = 3, 5, 7` の感度分析候補とする。

## 今後の改善候補

### epsilon-greedy の探索判定の修正
- 現状 `choose_action` の探索判定が `if np.random.randn() >= epsilon:` になっている。
- `np.random.randn()` は標準正規乱数のため、`epsilon=0.1` と比較すると greedy 率 ≈ 46%・random 率 ≈ 54% となり、本来意図する「ε=0.1 で稀に探索」とは大きく乖離している。
- 正しくは `if np.random.rand() < epsilon:` を探索（ランダム選択）、それ以外を greedy とすべき（`np.random.rand()` は一様乱数 `[0,1)`）。
- これはベースライン（EXP003）由来の挙動であり EXP002 の変更点（burn-in）ではないが、序盤の項目選択挙動に影響しうるため、初期性能（step 1-10）を議論する際は修正を検討する。
- 本実験では他手法との比較条件を揃える目的で現状の挙動を維持しているが、別実験で探索率を正した条件と比較すると効果を切り分けやすい。
- EXP001 の設計書にも同じ改善候補を記載済み（設計書間で整合）。
