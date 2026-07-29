# EXP029: DQN（ネットワークパラメータの正値制約なし）

## 目的

EXP027の実データ版とシミュレーション版を基準に、学習時の
`Apply_Positive_Constraint` を適用しない場合の影響を確認する。

## Notebook

- `EXP029/notebook/Train_and_test_DQN_on_the_real_responses.ipynb`
- `EXP029/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`
- `EXP029/notebook/Train_and_test_DQN_Param_Raw_MLE_on_the_simulated_bank.ipynb`
- `EXP029/notebook/plot_rmse_comparison_mle.ipynb`

## EXP027からの変更

- `Apply_Positive_Constraint` 関数を削除
- `loss.backward()` と `optimizer.step()` の間にあった
  `Apply_Positive_Constraint(eval_net)` 呼び出しを削除
- モデル保存先を `EXP029/models/` に変更
- 結果保存先を `EXP029/results/` に変更

これ以外のアルゴリズムと実験条件は、対応するEXP027 notebookと同一である。
正値制約を外すため、ネットワークの重み・バイアスには負値が許され、Q値も非負であることを
保証しない。

## DQN-Param-Raw MLE版

EXP023のDQN-Param-Rawを基準に、能力推定をEXP029のMLE実装へ置き換えた。
EXP023から維持する条件とMLEへの変更を区別する。

- 状態: `[theta_hat_MLE, t/L]`
- 項目特徴: バンク内で標準化した3PLパラメータ `[a, b, c]`
- Qネットワーク: 全候補項目で共有する `5 -> 64 -> 64 -> 1` MLP
- 報酬: 回答前の推定値における選択項目のFisher情報量
  `I_i(theta_hat_MLE)`
- TDターゲット: 次状態の残存項目のみを候補とし、終端では `Q(next)=0`
- MLE: `scipy.optimize.minimize_scalar` による `[-4,4]` の有界最適化
- 全正答・全誤答: EXP029と同じく、現在の推定値とバンクのdifficulty最大値・最小値の
  中点へ段階的に更新
- 初期推定値: 各受検者について `Uniform(-0.5, 0.5)`
- 学習theta: `Config.prior="normal"` では `N(0,1)`、`"uniform"` では
  `U(-3,3)`
- 既定値: gamma `0.1`、seed `42`。その他の学習条件はEXP023を維持
- テスト応答: テスト開始時に `Config.seed` でNumPy乱数を固定
- 正値重み制約: なし

保存名には `DQN_Param_Raw_MLE`、学習thetaのprior、gamma、seedを含め、
通常のDQNおよびEXP023のEAP版と区別する。

## 実データ版の実験条件

- データセット: `data/LNIRT_CredentialForm1/`
- 項目数: 170
- 訓練: training CSVから検証200人を除いた1,108人
- 検証: 200人（`seed=20260430` による固定分割）
- テスト: 328人
- テスト長: 40
- 能力推定: MLE
- 状態: 現在の能力推定値 `theta_hat`（1次元）
- 報酬: 各訓練受検者の参照 `true theta` における選択項目のFisher情報量
- discount factor: `gamma=0.5`
- epsilon-greedy: `epsilon=0.1`
- replay memory: 1,000
- batch size: 128
- target network更新間隔: 40 learning steps
- TDターゲットでは出題済み項目をaction maskで除外
- 終端transitionの `Q(next)` は0
- MLEおよび全正答・全誤答時の能力推定値は `[-4, 4]` にclip
- seed: `20260430`

## シミュレーション版の実験条件

- データセット: `data/uncorrelated_banks/item_bank_uncor_1.csv`
- 項目数: 500
- 訓練受検者: 1,000人（`N(0, 1)` からthetaを生成）
- 検証受検者: 200人
- テスト: `data/theta_true/theta_true_1.csv` の全受検者
- 応答: 3PLから項目ごとに独立生成
- テスト長: 40
- 能力推定: MLE
- 状態: 現在の能力推定値 `theta_hat`（1次元）
- 報酬: 訓練受検者の真のthetaにおける選択項目のFisher情報量
- discount factor: `gamma=0.9`
- epsilon-greedy: `epsilon=0.1`
- replay memory: 1,000
- batch size: 128
- target network更新間隔: 40 learning steps
- TDターゲットでは出題済み項目をaction maskで除外
- 終端transitionの `Q(next)` は0
- seed: `20260430`（テスト時のNumPy乱数を固定）

## 検証状況

既定値による全規模実験は未実行。縮小設定によるスモークテストで、実データ読込、
正値制約なしの学習、検証、テスト、モデル・records CSV・summary CSVの保存まで確認する。

シミュレーション版についても縮小設定で同じ経路をスモークテストする。

結果を解釈する際は、同じ `data/LNIRT_CredentialForm1/` 上のMFIと比較する。

## RMSE比較プロット

MLE推定に限定し、EXP029のDQNとEXP027のPython生成MFIを比較する。

- シミュレーションMFI:
  `EXP027/results/summary_uncor_1_MFI_MLE_python.csv`
- 実データMFI:
  `EXP027/results/summary_real_LNIRT_CredentialForm1_MFI_MLE.csv`
- シミュレーションDQN: 通常DQNとDQN-Param-Rawのそれぞれについて、
  normal/uniformともにgamma `0.1`で固定
- 実データDQN: `EXP029/results/` にある全gamma
- 出力:
  `EXP029/results/rmse_comparison_MLE_simulated_and_real.png`
- 実データMFI対DQN（gamma=0.3）単独図:
  `EXP029/results/rmse_comparison_MLE_real_gamma_0.3.png`

実データとシミュレーションではRMSEのスケールが異なるため、プロットのY軸は共有しない。
