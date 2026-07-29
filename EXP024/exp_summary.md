# EXP024: DQN-Param-Raw with Log Posterior Variance and Three Hidden Layers

## 目的

EXP023のDQN-Param-Rawに対して、EAP事後分散のlog値を状態へ追加し、共有Qネットワークの隠れ層を1層増やす。能力推定の不確実性と、状態・項目パラメータ間の非線形な関係をより表現できるか確認する。

## EXP023からの変更点

| 変更内容 | EXP023 | EXP024 |
|---|---|---|
| 状態 | `[theta_hat_EAP, t/L]` | `[theta_hat_EAP, log(post_var_EAP), t/L]` |
| 状態次元 | 2 | 3 |
| Qネットワーク入力 | 5次元 | 6次元 |
| 共有MLP | `5 -> 64 -> 64 -> 1` | `6 -> 64 -> 64 -> 64 -> 1` |

報酬、TDターゲット、項目特徴、EAP計算法、学習条件、データセットはEXP023から変更しない。

## 状態

時点`t`は次の項目を選択する直前とする。

$$
s_t=
\left[
\hat\theta_t,
\log\left(\operatorname{Var}(\theta\mid\mathcal D_t)\right),
\frac{t}{L}
\right]
$$

EAP事前分布は$N(0,1)$なので、初期状態は、

$$
s_0=[0,\log(1),0]=[0,0,0]
$$

とする。数値安定性のため、実装では、

$$
\log\left(\max(\operatorname{post\_var},10^{-10})\right)
$$

を入力する。

## 項目特徴

EXP023と同様に、`item_bank_uncor_1.csv`全体の平均・母標準偏差で標準化した3PL項目パラメータを使用する。

$$
x_i=[\tilde a_i,\tilde b_i,\tilde c_i]
$$

動的特徴量$P_i(\hat\theta_t)$や$I_i(\hat\theta_t)$は追加しないため、本実験もRaw版に分類する。

## Qネットワーク

$$
Q_\psi(s_t,x_i)=
f_\psi
\left(
[\hat\theta_t,\log(\operatorname{post\_var}_t),t/L,
\tilde a_i,\tilde b_i,\tilde c_i]
\right)
$$

共有MLPは、

$$
6\rightarrow64\rightarrow64\rightarrow64\rightarrow1
$$

とし、各隠れ層にReLUを使用する。最終層は線形出力で、重みの正値制約は使用しない。

## 報酬とTDターゲット

EXP023から変更しない。

$$
r_t=I_{i_t}(\hat\theta_t)
$$

$$
y_t=r_t+\gamma(1-d_t)
\max_{j\in\mathcal A_{t+1}}
Q_{\psi^-}(s_{t+1},x_j)
$$

次状態の最大Q値は残存項目だけから計算し、終端状態では将来Q値を0とする。

## 実験条件

| パラメータ | 値 |
|---|---|
| 使用バンク | `data/uncorrelated_banks/item_bank_uncor_1.csv` |
| 項目数 | 500 |
| 能力推定 | EAP（61点グリッド、範囲`[-4,4]`） |
| EAP事前分布 | `N(0,1)` |
| テスト長 | 40 |
| gamma | 0.1 |
| epsilon | 0.1 |
| replay capacity | 1000 |
| batch size | 128 |
| target network更新間隔 | 40 gradient updates |
| optimizer | Adam、学習率`1e-3` |
| training size | 1000 |
| validation size | 200 |
| validation interval | 50 episodes |
| seed | 42 |

## ファイル

- 実装Notebook: `EXP024/notebook/Train_and_test_DQN_Param_Raw_LogVar_3Layer_on_the_simulated_bank.ipynb`
- 同内容のPythonソース: `EXP024/notebook/Train_and_test_DQN_Param_Raw_LogVar_3Layer_on_the_simulated_bank.py`
- 比較Notebook: `EXP024/notebook/plot_rmse_comparison.ipynb`
- モデル: `EXP024/models/`
- 結果: `EXP024/results/`

## 主要結果

実験実行後に記入する。

| Test length | MFI（EAP、現行結果） | EXP023 gamma=0.1 | EXP024 gamma=0.1 |
|---:|---:|---:|---:|
| 10 | 0.504 | 0.500 | - |
| 20 | 0.372 | 0.378 | - |
| 30 | 0.309 | 0.317 | - |
| 40 | 0.271 | 0.282 | - |

評価時はEXP024単独のRMSEだけでなく、同じ項目バンク上のMFIおよびEXP023とstep 10、20、30、40で比較する。

なお、現行MFIは初期能力値に`Uniform(-0.5, 0.5)`、EXP023/024はEAP事前平均0を使用している。厳密な比較ではMFI側の初期値とEAP数値計算法も統一する必要がある。
