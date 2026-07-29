# EXP023: DQN-Param-Raw（EAP・単一非相関バンク）

## 目的

EXP020 の EAP 能力推定を維持し、項目 ID ごとに専用の Q 値を出力するネットワークを、3PL 項目パラメータを入力する共有 Q ネットワークへ置き換える。

本実験では、項目自身のパラメータを入力する効果だけを確認するため、動的な IRT 特徴量や残存項目集合の集約表現は加えない。

## EXP020からの主な変更点

| 変更内容 | EXP020 | EXP023 |
|---|---|---|
| 状態 | `[theta_hat_EAP]` | `[theta_hat_EAP, t/L]` |
| 行動の表現 | 項目 ID ごとの出力ユニット | 標準化した `[a, b, c]` |
| Q ネットワーク | `1 -> 50 -> 30 -> 500` | 共有 MLP `5 -> 64 -> 64 -> 1` |
| 初期能力推定値 | `Uniform(-0.5, 0.5)` | EAP 事前平均 `0` |
| 報酬 | `I_i(theta_true)` | `I_i(theta_hat_t)` |
| TDターゲットの候補 | 全項目（出題済み項目を含む） | 次状態の残存項目のみ |
| 終端処理 | 学習ループの現在位置に依存 | replay transitionの終端フラグを使用 |
| 正値重み制約 | あり（ただし更新前適用） | なし |
| eval/target初期化 | 独立 | 同一重み |
| 乱数seed | 固定なし | Configの `seed=42` |

EXP020から複数の条件が変わるため、EXP023とEXP020の差を「項目パラメータ入力だけの効果」とは解釈しない。EXP023は、ユーザー指定のDQN-Param定式化を満たす最初の実装として位置づける。

## 定式化

時点 `t` は次の項目を選択する直前を表し、最初の選択前を `t=0` とする。

### 状態

$$
s_t = \left[\hat\theta_t, \frac{t}{L}\right]
$$

初期状態は、EAP事前分布を $N(0,1)$ として、

$$
s_0=[0,0]
$$

とする。

### 項目特徴

元の項目パラメータを、学習に使用する単一バンク全体の平均と母標準偏差（`ddof=0`）で標準化する。

$$
x_i=
\left[
\frac{a_i-\mu_a}{\sigma_a},
\frac{b_i-\mu_b}{\sigma_b},
\frac{c_i-\mu_c}{\sigma_c}
\right]
$$

標準化はQネットワークへの入力だけに適用する。反応生成、EAP推定、Fisher情報量の計算には元の $a_i,b_i,c_i$ を使用する。

### Q値と行動選択

$$
Q_\psi(s_t,x_i)=
f_\psi\left(
[\hat\theta_t,t/L,\tilde a_i,\tilde b_i,\tilde c_i]
\right)
$$

$$
i_t=\underset{i\in\mathcal{A}_t}{\arg\max}\;Q_\psi(s_t,x_i)
$$

### 報酬

回答前の現在状態に含まれるEAP推定値でFisher情報量を計算する。

$$
r_t=I_{i_t}(\hat\theta_t)
$$

### TDターゲット

$$
y_t = r_t + \gamma(1-d_t)
\underset{j\in\mathcal{A}_{t+1}}{\max}
Q_{\psi^-}(s_{t+1},x_j)
$$

replay memoryには `state`, `action`, `reward`, `next_state`, `terminal`, `next_available` を保存する。

## 実験条件

| パラメータ | 値 |
|---|---|
| 使用バンク | `data/uncorrelated_banks/item_bank_uncor_1.csv` |
| 項目数 | 500 |
| 能力推定 | EAP（61点グリッド、範囲 `[-4,4]`） |
| EAP事前分布 | `N(0,1)` |
| テスト長 | 40 |
| gamma | 0.1 |
| epsilon | 0.1 |
| replay capacity | 1000 |
| batch size | 128 |
| target network更新間隔 | 40 gradient updates |
| optimizer | Adam、学習率 `1e-3` |
| training size | 1000 |
| validation size | 200 |
| validation interval | 50 episodes |
| seed | 42 |

テスト対象の真の能力値は、bank 1に対応する `data/theta_true/theta_true_1.csv` を使用する。

## ファイル

- 実装Notebook: `EXP023/notebook/Train_and_test_DQN_Param_Raw_on_the_simulated_bank.ipynb`
- 同内容のPythonソース: `EXP023/notebook/Train_and_test_DQN_Param_Raw_on_the_simulated_bank.py`
- 比較Notebook: `EXP023/notebook/plot_rmse_comparison.ipynb`
- モデル: `EXP023/models/`
- DQN結果: `EXP023/results/`
- 比較対象MFI: `EXP007/results/summary_uncor_1_MFI.csv`

モデルチェックポイントには、モデル重み、Config、項目パラメータの平均・標準偏差を保存する。

## 主要結果

実験実行後に記入する。

| Test length | MFI（EXP007、同一バンク） | EXP020 DQN（EAP） | EXP023 DQN-Param-Raw |
|---:|---:|---:|---:|
| 10 | 0.701 | 0.529 | - |
| 20 | 0.462 | 0.406 | - |
| 30 | 0.377 | 0.341 | - |
| 40 | 0.336 | 0.310 | - |

EXP023の評価では、DQN単独のRMSEではなく、同じ `item_bank_uncor_1.csv` に対するMFIとstep 10、20、30、40で比較する。
