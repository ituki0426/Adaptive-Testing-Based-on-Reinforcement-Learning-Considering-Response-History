# EXP031: training / validation TD loss logging

## 目的

`EXP027/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`を基準に、
学習時のstep別Bias・RMSE・MAE行列を表示する代わりに、training lossと
validation lossを簡潔に表示・保存する。

Notebooks：

- 固定epsilon版：`EXP031/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`
- 線形減衰版：`EXP031/notebook/Train_and_test_DQN_linear_epsilon_decay_on_the_simulated_banks.ipynb`

## EXP027からの変更

- 各validation間に行われたDQN更新のTD MSEを平均し、training lossとする。
- validation受検者の全ステップの遷移についてTD MSEを計算し、全受検者・
  全ステップで平均した値をvalidation lossとする。
- 学習中は次の形式の1行だけを表示し、step別の指標行列は表示しない。

  `subject: 50, training loss: ..., validation loss: ...`

- validationごとのlossを`EXP031/results/loss_history_*.csv`へ保存する。
- モデルとテスト結果は`EXP031/models/`および`EXP031/results/`へ保存する。

## epsilonの固定版と線形減衰版

固定版の学習時epsilonは、他のベースラインnotebookと同じく全training stepで
`epsilon=0.1`とする。model、records、summary、loss historyのファイル名も従来の
固定版と同じである。

線形減衰版は
`Train_and_test_DQN_linear_epsilon_decay_on_the_simulated_banks.ipynb`とする。
学習時epsilonの既定値は`epsilon_start=1.0`、`epsilon_end=0.1`である。

総environment step数は`training_size * test_length`として実行時に計算する。
既定設定では`1000 * 40 = 40000` stepとなり、通算step indexを
$s=0,\ldots,39999$として次式を用いる。

$$
\epsilon_s
= 1.0 + (0.1 - 1.0)\frac{s}{40000 - 1}
$$

このため、最初の学習actionではepsilonが厳密に1.0、最後の学習actionでは
厳密に0.1となる。validationとtestの項目選択は従来どおりgreedyであり、epsilonは
使用しない。

validationごとのloss historyには、その時点の`environment_steps`と`epsilon`も
保存する。固定epsilon版の成果物を上書きしないよう、model、records、summary、
loss historyのファイル名には`epsilon_linear_1.0_to_0.1`を含める。

validation TDターゲットは学習時と同じく、次状態で出題済み項目を候補から除外し、
終端遷移の将来Q値を0とする。

## 変更しない条件

- DQNのネットワーク、報酬、状態、行動マスク、replay buffer、optimizer
- 能力推定方法（ConfigでMLE/EAPを選択）
- データセット、項目数、受検者数、テスト長、seed
- validationの能力推定指標の計算
- step 7〜40の平均RMSEによるベストモデル選択
- 固定test thetaを用いたテスト方法と評価指標

lossは学習状況の表示・記録にのみ使用し、checkpoint選択には使用しない。

## Oracle MFI + MLE

通常のMFI + MLEとは別に、項目選択時のFisher情報量を各受検者の真の特性値で
計算するoracle版を実装する。

Notebook：

`EXP031/notebook/Test_MFI_oracle_MLE_on_the_simulated_bank.ipynb`

各stepでは、受検者 $i$ の真値 $\theta_{i,\mathrm{true}}$ において未出題項目の
Fisher情報量を比較し、最大の項目を選択する。回答生成、回答後のMLE推定、
Bias・RMSE・MAE・相関、累積Fisher情報量、seedは通常MFI版と共通とする。
真値は項目選択と累積Fisher情報量の計算に使用する。

通常MFIの結果を上書きしないよう、recordsとsummaryの出力名には
`MFI_oracle_MLE`を含める。

## 検証状況

構文、Ruff lint / format、縮小設定による学習・validation・testのスモークテストを
実施済みである。Oracle MFIについても、縮小設定で真値におけるFisher情報量の
降順に未出題項目が選択されることを確認済みである。既定設定による全規模実験は実施済みであり、
結果は「実験結果のまとめと考察」節に記載する。

## Oracle MFI と通常 MFI の比較結果

使用バンクは `data/uncorrelated_banks/`（bank 1、500項目）、受検者 5000 名、
テスト長 40、能力推定は MLE、seed は両条件で共通である。両条件とも RMSE 等の
評価は MLE 推定値 theta-hat に対して計算しており、真値は oracle 版の項目選択と
累積 Fisher 情報量の計算のみに使用している。

結果ファイル：

- 通常 MFI：`EXP031/results/summary_uncor_1_MFI_MLE_python.csv`
- Oracle MFI：`EXP031/results/summary_uncor_1_MFI_oracle_MLE_python.csv`

| step | RMSE 通常 | RMSE oracle | 差（oracle − 通常） | r 通常 | r oracle | 累積 FI 通常 | 累積 FI oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.297 | 1.678 | +0.381 | 0.475 | 0.058 | 0.327 | 0.477 |
| 3 | 1.035 | 1.684 | +0.649 | 0.638 | 0.405 | 0.827 | 1.327 |
| 5 | 0.876 | 1.274 | +0.398 | 0.729 | 0.579 | 1.400 | 2.115 |
| 10 | 0.641 | 0.691 | +0.050 | 0.840 | 0.823 | 2.954 | 3.950 |
| 20 | 0.424 | 0.413 | -0.010 | 0.922 | 0.925 | 6.152 | 7.271 |
| 30 | 0.339 | 0.337 | -0.002 | 0.948 | 0.948 | 9.215 | 10.286 |
| 40 | 0.291 | 0.293 | +0.003 | 0.961 | 0.960 | 12.093 | 13.076 |

### 観察

- 累積 Fisher 情報量は全 40 step で oracle が通常 MFI を上回る（step 40 で
  13.076 対 12.093、約 8% 増）。真値で選んでいるので当然であり、実装が意図
  どおり動いていることの確認になる。
- 一方 RMSE は step 1〜11 で oracle が明確に悪く、step 12〜34 でわずかに良く、
  step 35〜40 では再びほぼ同等かわずかに悪い。step 40 は 0.2932 対 0.2905 で
  通常 MFI の方が低い。
- 早期の悪化は推定の暴走による。step 3 で |theta-hat| > 3 に張り付く受検者の
  割合は oracle 4.9% に対し通常 MFI 0.0% である。oracle は theta-hat を無視して
  真値基準で選び続けるため、選択項目の正答確率が真値に対して常に約 0.67 で
  一定になり（step 1〜40 の平均 P=0.669、実測正答率 0.668）、初期 3 問が全問
  同一になる確率が高い（全問正答 30.8%、全問誤答 3.9%、計 34.7%）。この場合
  MLE が境界（±4）へ発散する。通常 MFI は theta-hat に追随するため自己補正が
  働き、step 1 正答後の次項目の P(theta_true) は 0.505、誤答後は 0.904 と
  大きく振れる。結果として初期 3 問が全問同一になる割合は 15.2%（全問正答
  14.9%、全問誤答 0.2%）にとどまる。
- step 1 で選ばれる項目は通常 MFI が 2 種類、oracle が 13 種類であり、oracle の
  方が受検者ごとに分散している。

### 解釈

EXP031 の DQN は報酬を `FI(item, theta_true)` で定義している。この報酬は
受検者の応答に依存しない決定的な値なので、未出題制約のもとで割引累積報酬を
最大化する最適方策は「真値における FI 上位項目を大きい順に選ぶ」ことに一致する。
つまり oracle MFI は、この報酬設計における DQN の性能上限（任意の gamma に対する
最適方策）である。

その上限を達成しても step 40 の RMSE は 0.293 で、通常 MFI の 0.291 を下回らない。
したがって、真値 Fisher 情報量を報酬とする限り、DQN をどれだけうまく学習させても
MFI に対する RMSE 優位は原理的に得られない。元論文 Table 1 の DQN 優位が
再現できていない原因の一つが報酬設計側にあることを示す結果である。

なお本比較は bank 1 の 1 回の実行に基づく。

## catRによる通常MFI + MLE

Python notebookとは別に、catRライブラリを用いて通常MFIで項目選択する
Rスクリプトを実装した。

Script：

`EXP031/Rscript/Test_MFI_MLE_on_the_simulated_bank_catR.R`

- 項目選択：`catR::nextItem(..., criterion = "MFI")`
- 応答生成：`catR::genPattern()`
- 能力推定：混合正誤パターンでは`catR::thetaEst(method = "ML")`、全問正答・
  全問誤答ではPython版と同じ中点更新
- 既定の評価対象：`data/uncorrelated_banks/item_bank_uncor_1.csv`、
  `data/theta_true/theta_true_1.csv`の全受検者、500項目、テスト長40
- seed：20260430
- 出力先：`EXP031/results/`
- 出力名：`records_uncor_1_MFI_MLE_catR.csv`、
  `summary_uncor_1_MFI_MLE_catR.csv`

recordsとsummaryの列、および真値における累積Fisher情報量の定義は通常MFIの
Python notebookに合わせている。ただし、catR/RとNumPyでは乱数生成器が異なるため、
seed番号を揃えても個々の初期推定値と応答系列はPython版とは一致しない。

`BANK_TYPE`、`BANK_ID`、`TEST_LENGTH`、`TESTING_SIZE`、`N_ITEMS`、`SEED`、
`THETA_CSV`、`OUTPUT_DIR`、`OUTPUT_SUFFIX`の各環境変数で条件を変更できる。

## 通常MFIとoracle MFIのプロット

通常MFIとoracle MFIのPython版summaryを比較する独立したnotebookを実装した。
bank 1〜5それぞれのsummaryに含まれる各stepのRMSEとCumRewardを算術平均する。
受検者を5バンク間で結合したpooled RMSEではない。

Notebook：

`EXP031/notebook/plot_mfi_vs_mfi_oracle.ipynb`

RMSEとCumRewardは別々のFigureとして描画し、次のファイルへ保存する。

- `EXP031/results/mfi_vs_mfi_oracle_mean_rmse_uncor_banks_1-5.png`
- `EXP031/results/mfi_vs_mfi_oracle_mean_cum_reward_uncor_banks_1-5.png`

また、step 10、20、30、40の両指標とoracle−通常MFIの差を
`EXP031/results/mfi_vs_mfi_oracle_mean_key_steps_uncor_banks_1-5.csv`へ保存する。

`EXP031/notebook/plot_cum_reward_comparison.ipynb`では、bank 1の通常MFIと
全gammaのDQNに加えてoracle MFIのCumRewardも同じFigureで比較する。
step 10、20、30、40の比較CSVにもoracle MFIの行を含める。

## Oracle MFIとの選択項目Fisher情報量差

Notebook：

`EXP031/notebook/analyze_selected_item_fi_diff_vs_oracle.ipynb`

Bank 1の通常MFI、Oracle MFI、固定epsilon版DQN（gamma=0、0.1、0.3、0.6、
0.9）のrecordsを受検者・stepで対応付け、各手法が選んだ項目のFisher情報量を
同じ受検者の真のthetaで再計算する。各受検者$i$、step$t$、手法$m$について、
次の符号付き差と絶対差を計算する。

$$
d_{i,t,m}
=I_{a^{\mathrm{oracle}}_{i,t}}(\theta_{i,\mathrm{true}})
-I_{a^m_{i,t}}(\theta_{i,\mathrm{true}}),
\qquad |d_{i,t,m}|
$$

手法ごとに既出項目が異なるため、符号付き差は負にもなり得る。Oracle MFIへの
近さには平均絶対差を用いる。DQNとMFIの直接比較には
`DQNの絶対差 - MFIの絶対差`を用い、負の値をDQNの方が近いと解釈する。

出力：

- `EXP031/results/selected_item_fi_diff_vs_oracle_uncor_1.csv`
- `EXP031/results/selected_item_fi_diff_vs_oracle_uncor_1.png`
- `EXP031/results/dqn_vs_mfi_oracle_fi_closeness_uncor_1.csv`
- `EXP031/results/dqn_vs_mfi_oracle_fi_closeness_uncor_1.png`
- `EXP031/results/selected_item_fi_diff_vs_oracle_key_steps_uncor_1.csv`

主要結果では、step 5で全gammaのDQNがMFIより小さい平均絶対差を示した。
gamma=0.1はstep 2〜10、gamma=0.3はstep 2〜8とstep 10でMFIより小さい。
一方、step 20以降は全gammaでDQNの方がMFIよりOracleから遠い。step 40の
平均絶対差はMFIが0.0290、DQNはgamma=0.1の0.0795が最小で、gamma=0.9の
0.1350が最大である。したがって、DQNがMFIよりOracleに近いFIの項目を選ぶ傾向は
テスト序盤の一部stepに限られ、終盤では確認できない。結果はbank 1の各1回の
実行に基づく。

## 選択頻度上位10項目の重複分析

Notebook：

`EXP031/notebook/analyze_top10_selected_item_overlap.ipynb`

Bank 1について、各stepで5,000人に選択された頻度が高い項目を手法ごとに上位
10個抽出する。固定epsilon版DQNの各gammaについて、通常MFIおよびOracle MFIの
上位項目集合との重複数とJaccard類似度を比較する。同じ選択人数の項目はitemIDの
昇順で順位を決める。ユニーク選択項目が10個未満のstepでは、存在する項目だけを
使用し、ゼロ回選択の項目による水増しは行わない。

DQNがどちらの手法に近いかは、次のJaccard類似度差で判定する。

$$
J(\mathrm{DQN},\mathrm{Oracle\ MFI})
-J(\mathrm{DQN},\mathrm{MFI})
$$

正ならOracle MFI、負なら通常MFI、0なら同程度とする。

出力：

- `EXP031/results/top10_selected_items_by_step_uncor_1.csv`
- `EXP031/results/dqn_top10_item_overlap_uncor_1.csv`
- `EXP031/results/dqn_top10_item_overlap_key_steps_uncor_1.csv`
- `EXP031/results/dqn_top10_item_jaccard_uncor_1.png`
- `EXP031/results/dqn_top10_item_jaccard_oracle_minus_mfi_uncor_1.png`

全40 stepの判定数は次のとおりである。

| gamma | MFIに近い | Oracle MFIに近い | 同程度 |
|---:|---:|---:|---:|
| 0.0 | 23 | 6 | 11 |
| 0.1 | 23 | 9 | 8 |
| 0.3 | 25 | 10 | 5 |
| 0.6 | 13 | 12 | 15 |
| 0.9 | 15 | 8 | 17 |

step 5では全gammaがOracle MFI側に近い。一方、step 10ではgamma=0のみOracle
MFI側、gamma=0.1、0.3、0.9はMFI側、gamma=0.6は同程度である。step 40では
gamma=0.3だけがMFI側で、他は両方との重複が同程度である。全体ではMFI側の
stepが多いが、Jaccard類似度そのものが低くstep間変動も大きい。なお280個の
手法・step集合のうち258個は10項目で、序盤の22個はユニーク選択項目が10個未満
である。結果はbank 1の各1回の実行に基づく。

## DQN with negative-MSE reward

Fisher情報量報酬版とは別に、回答後の推定値に対する負の二乗誤差を報酬とする
notebookを実装した。

Notebook：

`EXP031/notebook/Train_and_test_DQN_negative_MSE_reward_on_the_simulated_banks.ipynb`

受検者$i$のstep $t$における報酬は次のとおりである。

$$
r_{i,t}=-\left(\hat{\theta}_{i,t}-\theta_{i,\mathrm{true}}\right)^2
$$

真値はシミュレーション中の報酬計算と評価だけに使用し、DQNの状態は従来どおり
推定値theta-hatのみとする。負のTDターゲットを表現できるよう、元のFisher情報量
報酬版にあるネットワークパラメータの正値制約は適用しない。それ以外のネットワーク、
replay buffer、行動マスク、能力推定、validationによるcheckpoint選択は維持する。

評価結果では、MFIとの比較に使用する真値での累積Fisher情報量を`CumReward`、
負のMSEの累積値を`CumTaskReward`として分離する。model、records、summary、
loss historyの出力名には`negative_MSE_reward`を含め、従来版を上書きしない。

構文、Ruff lint / format、および縮小設定によるtraining・validation・testの
スモークテストを実施済みである。既定設定による全規模実験は実施済みであり、
結果は「実験結果のまとめと考察」節に記載する。

## DQN with squared-error-reduction reward

負のMSE報酬版とは別に、回答前後の二乗推定誤差の減少量を報酬とするnotebookを
実装した。

Notebook：

`EXP031/notebook/Train_and_test_DQN_error_reduction_reward_on_the_simulated_banks.ipynb`

受検者$i$のstep $t$における報酬は次のとおりである。

$$
r_{i,t}
=\left(\hat{\theta}_{i,t-1}-\theta_{i,\mathrm{true}}\right)^2
-\left(\hat{\theta}_{i,t}-\theta_{i,\mathrm{true}}\right)^2
$$

step 1の回答前推定値には各受検者の初期状態を用いる。真値は報酬計算と評価だけに
使用し、DQNの状態には含めない。報酬とQ値は負にもなり得るため、ネットワーク
パラメータの正値制約は適用しない。それ以外の学習・評価条件とRMSEによるcheckpoint
選択はFisher情報量報酬版に合わせる。

評価結果では、真値での累積Fisher情報量を`CumReward`、二乗誤差減少量の累積値を
`CumTaskReward`として分離する。model、records、summary、loss historyの出力名には
`error_reduction_reward`を含める。

構文、Ruff lint / format、および縮小設定によるtraining・validation・testの
スモークテストを実施済みである。`CumTaskReward`のstep間増分が、保存された推定値
から再計算した二乗誤差の減少量と一致することも確認済みである。既定設定による
全規模実験は実施済みであり、結果は「実験結果のまとめと考察」節に記載する。

## 実験結果のまとめと考察（bank uncor_1）

### 共通条件

- バンク：`data/uncorrelated_banks/item_bank_uncor_1.csv`（500項目）
- テスト受検者：5000名、テスト長 40、能力推定：MLE、seed：20260430
- DQN 学習：training_size=1000、validation_size=200、validation_interval=50、
  memory_capacity=1000、batch_size=128、lr=1e-3、target 更新 40 step
- 状態は theta-hat のみ（input_size=1）、ベストモデルは validation RMSE で選択
- MFI / oracle MFI と DQN は同一バンク・同一 seed・同一 test theta で比較している

### 全条件の RMSE と真値累積 Fisher 情報量

RMSE（低いほど良い）：

| 条件 | step 10 | step 20 | step 30 | step 40 |
|---|---:|---:|---:|---:|
| MFI (MLE) | 0.6412 | 0.4238 | 0.3389 | **0.2905** |
| Oracle MFI (MLE) | 0.6913 | 0.4135 | 0.3366 | 0.2932 |
| DQN FI報酬 gamma=0 | 0.6545 | 0.4556 | 0.3715 | 0.3359 |
| DQN FI報酬 gamma=0.1 | 0.6325 | 0.4468 | 0.3713 | 0.3247 |
| DQN FI報酬 gamma=0.3 | 0.6197 | 0.4429 | 0.3787 | 0.3428 |
| DQN FI報酬 gamma=0.6 | 0.6372 | 0.4485 | 0.3723 | 0.3239 |
| DQN FI報酬 gamma=0.9 | 0.6755 | 0.4745 | 0.3946 | 0.3574 |
| DQN FI報酬 gamma=0.3 + eps 線形減衰 | 0.6250 | 0.4380 | 0.3588 | 0.3161 |
| DQN 負のMSE報酬 gamma=0.1 | 0.8638 | 0.6065 | 0.5027 | 0.4394 |
| DQN 負のMSE報酬 gamma=0.3 | 0.9338 | 0.6378 | 0.5454 | 0.4632 |
| DQN 負のMSE報酬 gamma=1 | 0.9599 | 0.7027 | 0.5822 | 0.4998 |
| DQN 誤差減少報酬 gamma=0.1 | 0.7988 | 0.6203 | 0.5318 | 0.4606 |
| DQN 誤差減少報酬 gamma=0.3 | 0.8395 | 0.6202 | 0.5201 | 0.4545 |

真値における累積 Fisher 情報量 CumReward（高いほど良い）：

| 条件 | step 10 | step 20 | step 30 | step 40 |
|---|---:|---:|---:|---:|
| MFI (MLE) | 2.954 | 6.152 | 9.215 | 12.093 |
| Oracle MFI (MLE) | **3.950** | **7.271** | **10.286** | **13.076** |
| DQN FI報酬 gamma=0 | 3.067 | 5.933 | 8.540 | 10.574 |
| DQN FI報酬 gamma=0.1 | 2.999 | 6.023 | 8.632 | 10.965 |
| DQN FI報酬 gamma=0.3 | 3.115 | 5.883 | 8.414 | 10.446 |
| DQN FI報酬 gamma=0.6 | 2.990 | 5.835 | 8.327 | 10.290 |
| DQN FI報酬 gamma=0.9 | 2.842 | 5.200 | 7.179 | 8.680 |
| DQN FI報酬 gamma=0.3 + eps 線形減衰 | 3.003 | 6.054 | 8.746 | 10.929 |
| DQN 負のMSE報酬 gamma=0.1 | 1.616 | 3.157 | 4.593 | 5.926 |
| DQN 誤差減少報酬 gamma=0.3 | 1.597 | 3.022 | 4.334 | 5.621 |

全 13 条件のうち、step 40 の RMSE で MFI（0.2905）を下回るものは一つもない。

### 考察 1：Oracle MFI は FI 報酬の性能上限であり、その上限が MFI と同等である

報酬 `FI(item, theta_true)` は受検者の応答に依存しない決定的な値なので、未出題
制約のもとで（任意の gamma に対する）割引累積報酬を最大化する最適方策は
「真値における FI の大きい順に選ぶ」ことに一致する。すなわち oracle MFI は
FI 報酬 DQN の到達可能な性能上限である。

その oracle MFI は累積 FI で MFI を全 step 上回る（step 40 で 13.076 対 12.093、
+8.1%）にもかかわらず、step 40 の RMSE は 0.2932 で MFI の 0.2905 を下回らない。
5 バンク平均でも step 40 の差は −0.004 にとどまる。

したがって、**真値 Fisher 情報量を報酬とする限り、DQN をどれだけうまく学習させても
MFI に対する明確な RMSE 優位は原理的に得られない**。累積 FI を目的関数とした最適化
そのものが、この設定では RMSE の改善に結びつかない。元論文 Table 1 の DQN 優位が
再現できない原因の一つは、学習の巧拙ではなく報酬設計の側にある。

なお oracle MFI は step 1〜11 で RMSE が MFI より明確に悪い。真値に最適な項目を
最初から出すため正誤が 50:50 に近くなり、初期の全問正答・全問誤答パターンで MLE が
境界（±4）へ発散しやすいためである（step 3 で |theta-hat|>3 の割合は oracle 4.9%、
通常 MFI 0.0%）。通常 MFI は theta-hat に追随するため、誤答後は易しい項目へ移る
という自己補正が働く。

### 考察 2：DQN は累積 FI でも MFI を下回る（step 15 以降）

step 1〜14 では DQN（gamma=0, 0.1, 0.3）の累積 FI は MFI とほぼ同じかわずかに高い。
しかし step 15 前後で逆転し、以降は差が単調に開く。逆転が定着する step は
gamma=0.9 で 11、gamma=0.6 で 12、gamma=0/0.1/0.3 で 14〜15 である。

これは「DQN が短期の情報量を犠牲にして長期の利得を取っている」という説明が
成立しないことを意味する。DQN は長期でも負けている。step 40 の累積 FI は最良の
gamma=0.1 でも 10.965 で、MFI の 12.093 に対して −9.3% である。

gamma を大きくするほど累積 FI は単調に悪化する（gamma=0.1 で 10.97、0.9 で 8.68）。
長期報酬を重視する設定ほど結果が悪いのは、割引が学習を難しくしているだけで、
この問題に長期の信用割当がほとんど必要ないことを示す。実際 gamma=0（1 step 先だけ
見る）でも 10.574 で gamma=0.9 を大きく上回る。

### 考察 3：DQN の劣化は theta の裾に集中している

step 40 の RMSE を真値 theta の帯域別に分解すると次のようになる。

| 条件 | θ<−2 (n=121) | −2〜−1 (681) | −1〜1 (3433) | 1〜2 (659) | θ>2 (106) |
|---|---:|---:|---:|---:|---:|
| MFI | 0.393 | 0.319 | 0.278 | 0.292 | 0.351 |
| Oracle MFI | 0.415 | 0.329 | 0.279 | 0.291 | 0.353 |
| DQN FI報酬 gamma=0.1 | 0.695 | 0.377 | 0.285 | 0.322 | 0.490 |
| DQN FI報酬 gamma=0.9 | 0.628 | 0.425 | 0.318 | 0.379 | 0.492 |
| DQN eps線形減衰 gamma=0.3 | 0.518 | 0.362 | 0.286 | 0.352 | 0.393 |
| DQN 負のMSE報酬 gamma=0.1 | 0.717 | 0.526 | 0.400 | 0.454 | 0.536 |
| DQN 誤差減少報酬 gamma=0.3 | 0.666 | 0.541 | 0.419 | 0.469 | 0.562 |

中央帯（−1〜1、全体の 69%）では DQN gamma=0.1 は 0.285 で MFI の 0.278 とほぼ同等
である。差が出るのは裾で、θ<−2 では 0.695 対 0.393（+77%）、θ>2 では 0.490 対 0.351
（+40%）となる。全体 RMSE の差 0.0342 は、ほぼ全部この裾から来ている。

テスト全体で実際に使われた項目数も、FI 報酬 DQN は 82〜107 項目（500 中）に対し、
MFI は 215 項目、oracle MFI は 219 項目である。DQN は少数の項目に方策が潰れており、
裾の受検者に適した項目を出せていない。

これは既に特定済みの `Apply_Positive_Constraint`（全パラメータを clamp(min=0)）に
よる制約と整合する。全 weight/bias が非負かつ ReLU なので Q_a(theta-hat) は全項目で
theta-hat に対し単調非減少になる。一方ターゲットの FI(a, theta) は theta≒b_a に山を
持つ単峰関数なので、Q の順位が theta を追えず、argmax が少数項目に潰れる。裾で
崩壊するという観測パターンはこの機構の予測どおりである。

### 考察 4：epsilon 線形減衰は効くが、天井は動かない

gamma=0.3 で固定 epsilon=0.1 と線形減衰（1.0→0.1、40000 environment step で線形）を
比較すると次のようになる。

| 指標 | 固定 eps=0.1 | 線形減衰 1.0→0.1 | 差 |
|---|---:|---:|---:|
| step 40 RMSE | 0.3428 | 0.3161 | −0.0267 |
| step 30 RMSE | 0.3787 | 0.3588 | −0.0199 |
| step 40 累積 FI | 10.446 | 10.929 | +0.483 |
| 使用項目数 | 86 | 105 | +19 |
| θ<−2 の RMSE | — | 0.518 | （gamma=0.1 の 0.695 より改善） |

線形減衰は同 gamma の固定 eps に対して step 40 RMSE を 7.8% 改善し、全 gamma・
全 DQN 条件の中で最良（0.3161）となった。使用項目数と裾の RMSE も改善しており、
初期の強い探索が方策の項目カバレッジを広げるという想定どおりに働いている。

一方で、それでも MFI の 0.2905 には届かない（差 0.026）。学習曲線を見ても、
validation TD loss は 0.203→0.012 と減少しており、学習自体は安定している。
つまり**探索不足は真の律速ではなく副次要因**である。考察 1 と考察 3 が示すとおり、
上限は報酬設計（考察 1）とネットワークの表現制約（考察 3）の側にある。

### 考察 5：代替報酬（負の MSE / 二乗誤差減少）は大幅に悪化する

推定誤差を直接報酬にすれば RMSE が改善するという期待は、実験では成立しなかった。
step 40 RMSE は負の MSE 報酬で 0.439〜0.500、二乗誤差減少報酬で 0.455〜0.461 であり、
FI 報酬版の 0.316〜0.357 より明確に悪い。累積 FI も 5.1〜5.9 と、MFI の 12.09 の
半分以下である。

主因は報酬の分散である。FI 報酬は決定的だが、誤差ベースの報酬は 1 問の二値応答が
MLE 推定値を動かした結果に依存するため、同じ (state, action) に対する報酬が大きく
ばらつく。実際 validation TD loss は FI 報酬版が 0.010〜0.016 なのに対し、負の MSE
報酬版は 2.3〜3.8、誤差減少報酬版は 0.77〜1.7 で、2 桁大きい。training loss も
最後まで下がりきらず（誤差減少報酬 gamma=0.3 で 1.03→0.60）、方策が学習できて
いない。

gamma=1（割引なし）の負の MSE 報酬では TD loss が 1.7 → 8.5e9 へ発散した。
これは割引なしの bootstrapping で価値が発散する典型的な挙動であり、この設定で
gamma=1 は使えない。

なお、これら 2 条件では負の TD ターゲットを表現するために
`Apply_Positive_Constraint` を外している。実際、使用項目数は 122〜242 と FI 報酬版
（82〜107）より明確に多く、MFI の 215 に匹敵する条件もある。**制約を外すと項目
カバレッジは回復する**という考察 3 の機構の傍証になっている。しかし同時に報酬も
変えているため、この 2 条件からは「制約除去の効果」を分離できない。

### 結論

1. FI 報酬の理論上限（oracle MFI）ですら MFI に RMSE で勝てない。元論文 Table 1 の
   DQN 優位は、この報酬設計の下では学習の改善だけでは到達できない。
2. 現行 DQN は上限どころか MFI の累積 FI にも step 15 以降届いていない。原因は
   `Apply_Positive_Constraint` による Q の単調性強制で、劣化は theta の裾に集中する。
3. epsilon 線形減衰は同 gamma で RMSE を 7.8% 改善する有効な変更だが、MFI との差は
   埋まらない。探索不足は副次要因である。
4. 誤差ベースの報酬は分散が大きすぎて現行の学習設定では機能しない。gamma=1 は発散する。
5. 本節の結果は bank 1 の各 1 回の実行に基づく（MFI / oracle MFI のみ bank 1〜5）。

### 次に確認すべきこと

- **報酬・gamma・探索は据え置き、`Apply_Positive_Constraint` の除去だけを行った条件**を
  実行し、MFI と比較する。EXP031 の 2 つの代替報酬版は制約除去と報酬変更を同時に
  行っているため、制約除去単独の効果がまだ分離できていない。これは
  「論文本文準拠の修正」ではなく「公開コードのバグ相当の除去」として記録する。
- 制約除去 + epsilon 線形減衰の組み合わせ。現時点で有効性が確認できている 2 変更である。
- 誤差ベース報酬を残すなら、報酬の分散低減（複数応答での平均、報酬スケーリング、
  Huber loss）と gamma<1 の徹底が前提になる。
- MFI が MFI 自身の理論上限（oracle）とほぼ同じ RMSE になる以上、DQN が MFI に勝つには
  「累積 FI 以外の目的」が必要である。元論文がどの目的関数で優位を出したのかを
  改めて確認する。
