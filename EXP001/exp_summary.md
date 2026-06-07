# EXP001 設計書

## 実験名
Learnable Initial Hidden State DRQN

## 目的
DRQN の初期 RMSE 悪化に対して、LSTM の初期 hidden state / cell state を固定ゼロではなく学習可能パラメータに置き換えたときの改善効果を確認する。

この実験では、通常 DRQN と比較して「序盤の hidden state 構築不足」が主因かどうかを切り分ける。変更点は初期状態の扱いのみに限定し、入力表現、報酬、行動選択、評価指標はベースラインから変えない。

## 参照論文
- Oh Hyunwoo, 金子知適 (2018). *LSTM の初期状態の学習による DRQN の改善*. `papers/IPSJ-GPWS2018034.md`
- Hausknecht, M., & Stone, P. (2015). *Deep Recurrent Q-Learning for Partially Observable MDPs*. `papers/1507.06527v4.md`

## 背景
- 現行 DRQN は各受検者の 1 問目で `h_0 = 0, c_0 = 0` から開始する。
- 回答履歴が短い序盤では、LSTM が内部状態を十分に形成する前に項目選択を行うことになる。
- その結果、序盤の Q 値推定が不安定になり、`step 1-10` の RMSE が悪化している可能性がある。
- Oh & 金子 (2018) は DRQN において初期 hidden state を固定値ではなく学習対象とする改善案を扱っている。本実験ではその着想を、現在の IRT-CAT 用 DRQN 実装に対して最小限の差分で導入する。

## ベースライン
- ベースコードは `EXP003/notebook/Train and test DRQN on the simulated banks.ipynb` の simulated-bank DRQN。
- 入力は回答履歴のみ。
- 観測系列は `[START, o_1, o_2, ..., o_t]` の形で LSTM に入力する。
- 報酬は真の `theta` における Fisher 情報量 `FI(item, theta_true)`。
- 評価指標は Bias, RMSE, MAE。
- テスト長は 40 問、アイテムバンクは 200 項目。

## 実装対象ファイル
- 実験コード: [EXP001/src/Train and test DRQN on the simulated banks.py](/C:/Users/tatib/Adaptive%20Testing%20Based%20on%20Reinforcement%20Learning%20Considering%20Response%20History/EXP001/src/Train%20and%20test%20DRQN%20on%20the%20simulated%20banks.py)
- 出力設計書: [EXP001/exp_summary.md](/C:/Users/tatib/Adaptive%20Testing%20Based%20on%20Reinforcement%20Learning%20Considering%20Response%20History/EXP001/exp_summary.md)

## 実装方針
通常 DRQN と比較して、変更点をモデルの初期状態に限定する。

- 新規モデル `LearnableInitDRQN` を実験ファイル内で定義する。
- `init_h`, `init_c` を `nn.Parameter` としてモデル内部に持たせる。
- `init_hidden(batch_size)` はゼロテンソルを生成せず、学習済み `init_h`, `init_c` を `repeat` して返す。
- LSTM 本体、埋め込み層、出力層の構成は通常 DRQN と同じに保つ。
- replay memory の形式、TD ターゲットの作り方、validation の選抜基準、test 時の推論手順はベースラインから変えない。

この設計により、性能差が出た場合に「初期 hidden state の学習可否」が主因として解釈しやすくなる。

## モデル設計の詳細

### 1. クラス構成
`LearnableInitDRQN` は以下のモジュールを持つ。

- `embed`: 応答トークン `{0, 1, START}` を埋め込む `nn.Embedding(3, embed_dim)`
- `lstm`: `nn.LSTM(embed_dim, lstm_hidden, batch_first=True)`
- `out`: Q 値を出力する `nn.Linear(lstm_hidden, action_space)`
- `dropout`: ベースラインと同じ Dropout
- `init_h`: 形状 `(1, 1, lstm_hidden)` の学習可能初期 hidden state
- `init_c`: 形状 `(1, 1, lstm_hidden)` の学習可能初期 cell state

### 2. `forward`
`forward(resps, hidden=None)` の処理は通常 DRQN と同じ。

- `resps` を埋め込みに通す
- Dropout を適用
- LSTM で系列処理
- 出力系列に Dropout を適用
- 各時点の Q 値を `out` で計算

つまり、初期状態の供給方法だけを変え、系列処理本体は変えていない。

### 3. `init_hidden`
通常 DRQN はここでゼロテンソルを返すが、本実験では以下の形に変更した。

- `self.init_h.repeat(1, batch_size, 1)`
- `self.init_c.repeat(1, batch_size, 1)`

これにより、各バッチ・各受検者は同一の学習済み初期状態から開始する。初期状態は個人別ではなく、方策全体に共通なパラメータである。

### 4. 初期化
`initialize()` では以下を行う。

- `Linear` 重み: Kaiming 初期化
- `Linear` バイアス: 0 初期化
- `LSTM` 重み: Kaiming 初期化
- `LSTM` バイアス: 0 初期化
- `Embedding` 重み: 正規分布初期化
- `init_h`, `init_c`: 0 初期化

ここで重要なのは、`init_h`, `init_c` を「ゼロ固定」ではなく「ゼロから学習開始」にしている点である。

## 学習ループの詳細

### 1. エピソード生成
1 受検者ごとに 40 問のエピソードを生成する。

- 最初の入力は `START_TOKEN`
- hidden state は `eval_net.init_hidden(1)` で初期化
- 各ステップで epsilon-greedy により未使用項目から 1 問選択
- 応答は `RESPOND` でシミュレーション
- 報酬は `FI(item_bank[action], theta_true)`
- theta 推定は既存ロジックに従い、全問正答・全問誤答時は補助更新、それ以外は `MLE`

この部分はベースラインと同一であり、EXP001 では hidden の初期値だけが違う。

### 2. replay memory
各エピソードについて以下を保存する。

- `resps`: 長さ `T+1` の応答系列。先頭に `START_TOKEN` を含む
- `actions`: 長さ `T` の選択項目系列
- `rewards`: 長さ `T` の報酬系列

この形式もベースラインと同じ。

### 3. ミニバッチ学習
replay memory からエピソードをサンプリングし、系列全体を一括で LSTM に通す。

- `q_full_eval = eval_net(resps_t, eval_net.init_hidden(batch_size))`
- `q_full_target = target_net(resps_t, target_net.init_hidden(batch_size))`
- `q_eval` は各時点で実際に選ばれた action の Q 値
- `q_next` は次時点で未使用項目のみを候補とした最大 Q 値
- 最終時点は terminal としてブートストラップしない

損失は通常の MSE による TD 誤差で計算する。

**重要**: 学習時にも `init_hidden(batch_size)` で LSTM の初期状態を明示的に与える。これを省略して `hidden=None` で呼ぶと LSTM はゼロ初期状態を使い、`init_h` / `init_c` が損失に寄与せず勾配が流れない。その場合 `init_h` / `init_c` はゼロ初期化のまま更新されず、本実験の「初期状態の学習」が成立しない（ベースラインと同一挙動になる）。validation・test も同じ `init_hidden` を使うため、学習された初期状態が一貫して反映される。

### 4. 最適化
- Optimizer は `Adam`
- 勾配クリッピング `max_norm=1.0`
- 更新後に出力層のみ正値制約を適用

正値制約は `out.weight`, `out.bias` のみに適用し、LSTM や埋め込みには掛けない。これは「Q 値は累積情報量の期待値なので非負」という既存実装の考え方を維持しつつ、内部表現まで過度に制約しないためである。

## 推論時の処理

### validation
- 各 validation タイミングで `validation_size=200` 人を使って step 別 Bias/RMSE/MAE を計算する。
- 比較基準は `step 7-40` の平均性能。
- `abs(Bias)` が改善し、かつ `RMSE` と `MAE` が両方改善した場合に best model を更新する。

### test
- `theta_true` 全体に対して 40 問の CAT を実行する。
- 各ステップで推定 `theta` を更新し、Bias/RMSE/MAE を集計する。
- 出力は通常 DRQN と比較しやすい CSV 形式で保存する。

## 出力ファイル

### モデル
- `EXP001/models/drqn_learnable_init_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.pt`

保存しているのは `state_dict`。通常 DRQN とはクラス名が異なるため、再読込時は `LearnableInitDRQN` を明示的に生成してから `load_state_dict` する前提である。

### 結果 CSV
- `records_*.csv`
  各受検者、各ステップの `itemID`, `resp`, `theta_est`, `bias` を保存
- `summary_*.csv`
  各ステップの `Bias`, `RMSE`, `MAE` を保存

ファイル名には `learnable_init` を含め、通常 DRQN と識別できるようにしている。

## 通常 DRQN との差分

### 変更した点
- モデルクラスを `DRQN` から `LearnableInitDRQN` に変更
- `init_hidden()` がゼロ初期化ではなく学習パラメータを返す
- モデル保存名を `learnable_init` 付きに変更

### 変更していない点
- 応答のみを入力する DRQN という枠組み
- 報酬定義
- replay memory の形式
- sequence 全長学習
- epsilon-greedy 方策
- validation と test の評価手順

## 期待する効果
- 序盤における hidden state 形成を補助し、`step 1-10` の RMSE を下げる可能性がある
- 後半性能は大きく変えず、主に初期性能を改善することを狙う

## 想定されるリスク
- 初期状態が一部の典型パターンに過適応し、後半の一般化を損なう可能性がある
- `init_h`, `init_c` が強くなりすぎると、実際の応答履歴より初期バイアスに引っ張られる可能性がある
- 初期 RMSE が改善しても、40 問時点の RMSE が悪化するなら採用しにくい

## 比較時の確認項目
- `step 1-10` の RMSE が通常 DRQN より改善しているか
- `step 40` の RMSE が維持されているか
- Bias が極端にずれていないか
- validation での採択モデルが不安定に入れ替わっていないか

## 今後の改善候補

### epsilon-greedy の探索判定の修正
- 現状 `choose_action` の探索判定が `if np.random.randn() >= epsilon:` になっている。
- `np.random.randn()` は標準正規乱数のため、`epsilon=0.1` と比較すると greedy 率 ≈ 46%・random 率 ≈ 54% となり、本来意図する「ε=0.1 で稀に探索」とは大きく乖離している。
- 正しくは `if np.random.rand() < epsilon:` を探索（ランダム選択）、それ以外を greedy とすべき（`np.random.rand()` は一様乱数 `[0,1)`）。
- これはベースライン由来の挙動であり EXP001 の変更点ではないが、序盤の項目選択挙動に影響しうるため、初期性能（step 1-10）を議論する際は修正を検討する。
- 本実験では他手法との比較条件を揃える目的で現状の挙動を維持しているが、別実験で探索率を正した条件と比較すると効果を切り分けやすい。

## 再現時の実行上の注意
- 乱数 seed は `Config.random_seed` で固定している
- ベースライン比較時は `bank_type`, `bank_id`, `prior`, `gamma`, `training_size` を通常 DRQN と揃える
- 本実験は simulated bank 用の実装のみ追加している
- 実データ版や notebook 版にはまだ同変更を反映していない
