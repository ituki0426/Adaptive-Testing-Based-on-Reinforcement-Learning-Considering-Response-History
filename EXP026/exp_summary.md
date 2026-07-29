# EXP026: Double DQN-Param-Raw（EAP・単一非相関バンク）

## 目的

EXP023のDQN-Param-RawをDouble DQNへ拡張する。500個の候補項目から最大Q値を選ぶ通常DQNで生じ得る過大評価を抑え、項目選択とRMSEが改善するか確認する。

## EXP023からの変更点

変更点はTDターゲットにおける次行動の選択・評価方法だけである。

| 処理 | EXP023（通常DQN） | EXP026（Double DQN） |
|---|---|---|
| 次行動の選択 | target network | eval network |
| 選択した次行動の評価 | target network | target network |

状態、項目特徴、ネットワーク、報酬、gamma、replay memory、残存項目マスク、EAP推定、学習人数、validation方式はEXP023から変更しない。

## 状態と項目特徴

$$
s_t=[\hat\theta_t,t/L]
$$

$$
x_i=[\tilde a_i,\tilde b_i,\tilde c_i]
$$

項目パラメータは`item_bank_uncor_1.csv`全体の平均・母標準偏差で標準化する。

## Qネットワーク

EXP023と同じ共有MLPを使用する。

$$
Q_\psi(s_t,x_i)=
f_\psi([\hat\theta_t,t/L,\tilde a_i,\tilde b_i,\tilde c_i])
$$

$$
5\rightarrow64\rightarrow64\rightarrow1
$$

## 報酬

EXP023と同じく、回答前のEAP推定値におけるFisher情報量を使用する。

$$
r_t=I_{i_t}(\hat\theta_t)
$$

## Double DQN TDターゲット

次状態で残っている項目だけを候補として、eval networkで次行動を選択する。

$$
a^*_{t+1}
=
\underset{j\in\mathcal A_{t+1}}{\arg\max}
Q_\psi(s_{t+1},x_j)
$$

その行動のQ値をtarget networkで評価する。

$$
\boxed{
y_t
=
r_t
+
\gamma(1-d_t)
Q_{\psi^-}(s_{t+1},x_{a^*_{t+1}})
}
$$

終端状態では将来Q値を0とする。eval networkとtarget networkの両方で、出題済み項目が次行動として選ばれないよう、eval networkのargmax前に残存項目マスクを適用する。

## 実験条件

| パラメータ | 値 |
|---|---|
| 使用バンク | `data/uncorrelated_banks/item_bank_uncor_1.csv` |
| 項目数 | 500 |
| 能力推定 | EAP（61点グリッド、範囲`[-4,4]`） |
| EAP事前分布 | `N(0,1)` |
| 初期状態 | `[0,0]` |
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

- 実装Notebook: `EXP026/notebook/Train_and_test_Double_DQN_Param_Raw_on_the_simulated_bank.ipynb`
- 同内容のPythonソース: `EXP026/notebook/Train_and_test_Double_DQN_Param_Raw_on_the_simulated_bank.py`
- 比較Notebook: `EXP026/notebook/plot_rmse_comparison.ipynb`
- モデル: `EXP026/models/`
- 結果: `EXP026/results/`

## 主要結果

実験実行後に記入する。

| Test length | MFI（EAP、現行結果） | EXP023 DQN gamma=0.1 | EXP026 Double DQN gamma=0.1 |
|---:|---:|---:|---:|
| 10 | 0.504 | 0.500 | - |
| 20 | 0.372 | 0.378 | - |
| 30 | 0.309 | 0.317 | - |
| 40 | 0.271 | 0.282 | - |

同じ項目バンク上のMFIおよびEXP023と、step 10、20、30、40のRMSEを比較する。

EXP023との効果分離を優先するため、各validation時に受検者と回答を再生成する既存方式も変更していない。このvalidation RMSEにはMonte Carlo変動が含まれるため、最終的な性能判断ではtest RMSEと複数seedも確認する必要がある。
