# EXP025: DQN-Param with Log Posterior Variance Reduction Reward

## 目的

EXP024の状態表現とネットワーク構造を維持し、報酬をFisher情報量からlog posterior varianceの減少量へ変更する。MFIと同じ即時Fisher情報量を模倣するのではなく、EAP推定の不確実性低下を直接最適化することでRMSEが改善するか確認する。

## EXP024からの変更点

| 変更内容 | EXP024 | EXP025 |
|---|---|---|
| 報酬 | `I_i(theta_hat_t)` | `log(post_var_t) - log(post_var_{t+1})` |
| gamma | 0.1 | 1.0 |

状態、項目特徴、Qネットワーク、EAP推定、replay memory、TDターゲットの残存項目マスク、学習人数などはEXP024から変更しない。

## 状態

$$
s_t=
\left[
\hat\theta_t,
\log\left(\operatorname{Var}(\theta\mid\mathcal D_t)\right),
\frac{t}{L}
\right]
$$

初期状態は、EAP事前分布$N(0,1)$より、

$$
s_0=[0,0,0]
$$

とする。分散のlog変換では、数値安定性のため下限$10^{-10}$を適用する。

## 報酬

項目$i_t$への回答を観測し、EAP事後分布を更新した後に報酬を計算する。

$$
\boxed{
r_t=
\log\left(\operatorname{post\_var}_t\right)
-
\log\left(\operatorname{post\_var}_{t+1}\right)
}
$$

実装上は、状態の第2要素を使って、

```python
reward = float(state[1] - next_state[1])
```

と計算する。予想外の回答によって事後分散が一時的に増加した場合、報酬は負になり得る。

## gamma

$$
\gamma=1.0
$$

有限長CATで終端処理を行うため、割引なしの累積報酬を使用する。$gamma=1$では、累積報酬が、

$$
\sum_{t=0}^{L-1}r_t
=
\log\left(\operatorname{post\_var}_0\right)
-
\log\left(\operatorname{post\_var}_L\right)
$$

とtelescopingし、最終時点のposterior variance最小化に対応する。

## Qネットワーク

EXP024から変更しない。

$$
Q_\psi(s_t,x_i)=
f_\psi
\left(
[\hat\theta_t,\log(\operatorname{post\_var}_t),t/L,
\tilde a_i,\tilde b_i,\tilde c_i]
\right)
$$

ネットワーク構造は、

$$
6\rightarrow64\rightarrow64\rightarrow64\rightarrow1
$$

である。

## TDターゲット

$$
y_t=
r_t+(1-d_t)
\max_{j\in\mathcal A_{t+1}}
Q_{\psi^-}(s_{t+1},x_j)
$$

次状態では出題済み項目をマスクし、終端状態では将来Q値を0とする。

## 実験条件

| パラメータ | 値 |
|---|---|
| 使用バンク | `data/uncorrelated_banks/item_bank_uncor_1.csv` |
| 項目数 | 500 |
| 能力推定 | EAP（61点グリッド、範囲`[-4,4]`） |
| EAP事前分布 | `N(0,1)` |
| 状態 | `[theta_hat, log(post_var), t/L]` |
| 項目特徴 | 標準化した`[a,b,c]` |
| ネットワーク | `6 -> 64 -> 64 -> 64 -> 1` |
| テスト長 | 40 |
| gamma | 1.0 |
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

- 実装Notebook: `EXP025/notebook/Train_and_test_DQN_Param_LogVar_Reward_on_the_simulated_bank.ipynb`
- 同内容のPythonソース: `EXP025/notebook/Train_and_test_DQN_Param_LogVar_Reward_on_the_simulated_bank.py`
- 比較Notebook: `EXP025/notebook/plot_rmse_comparison.ipynb`
- モデル: `EXP025/models/`
- 結果: `EXP025/results/`

## 主要結果

実験実行後に記入する。

| Test length | MFI（EAP、現行結果） | EXP023 gamma=0.1 | EXP024 gamma=0.1 | EXP025 gamma=1.0 |
|---:|---:|---:|---:|---:|
| 10 | 0.504 | 0.500 | - | - |
| 20 | 0.372 | 0.378 | - | - |
| 30 | 0.309 | 0.317 | - | - |
| 40 | 0.271 | 0.282 | - | - |

評価では、同じ項目バンク上のMFI、FI報酬を使うEXP024、log variance減少報酬を使うEXP025をstep 10、20、30、40で比較する。

現行MFIとEXP025では初期能力値とEAP数値計算法が完全には統一されていないため、厳密なMFI優劣の判断には比較条件の統一も必要である。
