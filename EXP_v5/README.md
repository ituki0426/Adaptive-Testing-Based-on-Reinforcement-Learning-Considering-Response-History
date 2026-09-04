# 実験設定

## 1.基本設定

- 潜在次元数：2
- IRTモデル：2PL
- 項目数：150
- テスト長:40
- 比較手法：DQN vs D-optimality vs Random baseline
- 能力推定方法：MAP推定
- 能力推定の事前分布：$N(\mathbf{0},\mathbf{I}_2)$（すべての能力相関条件で共通）

項目反応確率は、

$$
P(Y_{ij}=1 \mid \theta_i)
=
\frac{1}
{1+\exp\left[-\left(a_{j1}\theta_{i1}+a_{j2}\theta_{i2}-b_j\right)\right]}
$$

とする。

ここで、

$$
\theta_i
=
(\theta_{i1},\theta_{i2})^\top
$$

は受検者 $i$ の2次元能力値、

$$
\mathbf{a}_j
=
(a_{j1},a_{j2})
$$

は項目 $j$ の識別力ベクトル、$b_j$ は項目難易度である。

能力推定には、次の共通の事前分布を用いたMAP推定を採用する。

$$
\boldsymbol{\theta}
\sim
N(\mathbf{0},\boldsymbol{\Sigma}_0),
\qquad
\boldsymbol{\Sigma}_0=\mathbf{I}_2
$$

回答済み項目に対する対数尤度を $\ell_t(\boldsymbol{\theta})$ とすると、$t$ 問終了時点の能力推定値は

$$
\hat{\boldsymbol{\theta}}_t
=
\arg\max_{\boldsymbol{\theta}}
\left[
\ell_t(\boldsymbol{\theta})
-
\frac{1}{2}
\boldsymbol{\theta}^{\top}
\boldsymbol{\Sigma}_0^{-1}
\boldsymbol{\theta}
\right]
$$

とする。開始時には回答データがないため、事前平均に基づき

$$
\hat{\boldsymbol{\theta}}_0=\mathbf{0}
$$

とする。

真の能力値を生成する母集団分布では $\rho\in\{0,0.3,0.6\}$ を使用するが、MAP推定の事前分布はすべての条件で $N(\mathbf{0},\mathbf{I}_2)$ に固定する。したがって、真の $\rho$ はDQNおよびD-optimalityのどちらにも既知情報として与えず、受検者の生成条件としてのみ用いる。

## 2.パラメータ生成

各パラメータは乱数によって生成する。

| パラメータ | 記号・次元 | 生成方法 | 生成例 |
|---|---|---|---|
| 能力値 | $\boldsymbol{\theta}_i=(\theta_{i1},\theta_{i2})^\top$ | $\boldsymbol{\theta}_i \sim MVN\left(\begin{pmatrix}0\\0\end{pmatrix},\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}\right)$, $\rho \in \{0,0.3,0.6\}$ | $\boldsymbol{\theta}_1=(0.8,-0.3)^\top$, $\boldsymbol{\theta}_2=(-1.1,-0.7)^\top$, $\boldsymbol{\theta}_3=(0.2,1.4)^\top$ |
| 識別力 | $\mathbf{a}_j=(a_{j1},a_{j2})$ | $a_{j1}\sim U(0.5,2.0)$, $a_{j2}\sim U(0.5,2.0)$ | $\mathbf{a}_1=(1.4,0.8)$, $\mathbf{a}_2=(0.7,1.6)$, $\mathbf{a}_3=(1.8,1.1)$ |
| 難易度 | $b_j$ | $b_j\sim U(-3,3)$ | $b_1=-1.8$, $b_2=0.4$, $b_3=2.1$ |

能力値については、能力相関 $\rho$ を

$$
\rho \in \{0, 0.3, 0.6\}
$$

と変化させ、能力間相関の違いが項目選択性能に与える影響を調べる。

## 3.比較手法

D-optimality(MFIの代わり)、DQN、ランダムベースラインを比較する。

### 3.1. D-optimality

D-optimalityは、多次元CATにおいてフィッシャー情報量行列の行列式を最大化する項目選択法である。

すでに出題済みの項目集合を $S_{t-1}$ とすると、項目から得られるフィッシャー情報量行列の和は

$$
\mathbf{I}_{S_{t-1}}(\boldsymbol{\theta})
=
\sum_{j \in S_{t-1}}
\mathbf{I}_j(\boldsymbol{\theta})
$$

となる。

2次元2PLにおける項目 $j$ のフィッシャー情報量行列は、

$$
\mathbf{I}_j(\boldsymbol{\theta})
=
P_j(\boldsymbol{\theta})
\left(1-P_j(\boldsymbol{\theta})\right)
\mathbf{a}_j\mathbf{a}_j^\top
$$

であり、具体的には

$$
\mathbf{I}_j(\boldsymbol{\theta})
=
P_j(1-P_j)
\begin{pmatrix}
a_{j1}^2 & a_{j1}a_{j2} \\
a_{j1}a_{j2} & a_{j2}^2
\end{pmatrix}
$$

となる。ただし、1項目の情報行列は階数1であるため、D-optimalityの項目選択には、項目情報だけでなくMAP推定の事前情報行列 $\boldsymbol{\Sigma}_0^{-1}$ を含む次の行列を用いる。

$$
\mathbf{J}_{t-1}(\hat{\boldsymbol{\theta}}_{t-1})
=
\boldsymbol{\Sigma}_0^{-1}
+
\sum_{k\in S_{t-1}}
\mathbf{I}_k(\hat{\boldsymbol{\theta}}_{t-1})
$$

本実験では $\boldsymbol{\Sigma}_0=\mathbf{I}_2$ であるため、開始時の行列は

$$
\mathbf{J}_0=\boldsymbol{\Sigma}_0^{-1}=\mathbf{I}_2
$$

となる。

D-optimalityでは、候補項目 $j$ を追加した場合の

$$
\mathbf{J}_{t-1}(\hat{\boldsymbol{\theta}}_{t-1})
+
\mathbf{I}_j(\hat{\boldsymbol{\theta}}_{t-1})
$$

の行列式を計算し、

$$
j_t
=
\arg\max_{j \notin S_{t-1}}
\det
\left[
\mathbf{J}_{t-1}(\hat{\boldsymbol{\theta}}_{t-1})
+
\mathbf{I}_j(\hat{\boldsymbol{\theta}}_{t-1})
\right]
$$

となる項目を次の出題項目として選択する。

したがって、

$$
\text{D-optimality}
=
\text{事前情報を含む情報行列の行列式を最大化する項目選択法}
$$

と整理できる。

### 3.2. DQN

DQNへの入力状態は、現在の2次元能力推定値

$$
s_t
=
\hat{\boldsymbol{\theta}}_t
=
(\hat{\theta}_{1,t}, \hat{\theta}_{2,t})^\top
$$

とする。ここで、$\hat{\boldsymbol{\theta}}_t$ は上で定義したMAP推定値である。

本実験では、既存の1次元DQNからの変更を最小限にするため、Q-networkをそのまま2次元入力・150項目出力へ拡張する。ネットワーク構造は

$$
2 \rightarrow 50 \rightarrow 30 \rightarrow 150
$$

とし、各層の役割は次のとおりとする。

- 入力層：2次元のMAP推定値 $(\hat{\theta}_{1,t},\hat{\theta}_{2,t})$
- 第1隠れ層：50ユニット、ReLU活性化関数
- 第2隠れ層：30ユニット、ReLU活性化関数
- 出力層：150項目それぞれに対応するQ値

online networkとtarget networkには同一のネットワーク構造を使用する。出力値は各項目を選択した場合のQ値であり、確率ではないため、出力層にはsoftmaxを使用しない。

すでに出題した項目が再び選択されることを防ぐため、行動選択時には出題済み項目のQ値を $-\infty$ にマスクし、未出題項目だけをargmaxの候補とする。TDターゲットの次状態における最大Q値を計算する場合も、同様に次状態で出題済みの項目を $-\infty$ にマスクする。

D-optimalityに近い目的をDQNに学習させるため、フィッシャー情報量行列の log-determinant の増加量を報酬とする。

$$
r_t
=
\log \det \mathbf{J}_t
-
\log \det \mathbf{J}_{t-1}
$$

ここで、

$$
\mathbf{J}_t
=
\boldsymbol{\Sigma}_0^{-1}
+
\sum_{j \in S_t}
\mathbf{I}_j(\hat{\boldsymbol{\theta}}_t)
$$

である。

この報酬は、

**1問出題したことによって、テスト全体の多次元情報量がどれだけ増加したか**

を表す。

#### 3.2.1. Double DQN版

通常DQNとの比較用に、TDターゲットだけをDouble DQNへ変更したNotebookを
`EXP_v5/src/Train_and_test_Double_DQN_MAP_on_the_simulated_bank.ipynb`
として実装する。次状態の行動はonline networkで選択し、その行動のQ値はtarget
networkで評価する。

$$
a^*
=
\mathop{\arg\max}_{a \in \mathcal{A}(s_{t+1})}
Q_{\mathrm{online}}(s_{t+1},a),
$$

$$
y_t
=
r_t
+
\gamma
Q_{\mathrm{target}}(s_{t+1},a^*).
$$

$\mathcal{A}(s_{t+1})$ は次状態で未出題の項目集合であり、online networkによる
argmaxとtarget networkによる評価の両方で出題済み項目をマスクする。終端遷移では
将来Q値を0とする。ネットワーク、報酬、MAP推定、replay buffer、seed、validation
によるcheckpoint選択は通常DQN版と共通とし、出力名には`Double_DQN_MAP`を含めて
通常DQNの結果と区別する。

#### 3.2.2. RMSE減少量報酬版

log-determinant増分報酬との比較用に、通常DQNの報酬だけを各受検者の
推定誤差の減少量へ変更したNotebookを
`EXP_v5/src/Train_and_test_DQN_MAP_RMSE_reward_on_the_simulated_bank.ipynb`
として実装する。受検者 $i$ のステップ $t$ における距離を

$$
d_{i,t}
=
\sqrt{
\frac{1}{2}
\left\|
\hat{\boldsymbol{\theta}}_{i,t}
-
\boldsymbol{\theta}_{i,\mathrm{true}}
\right\|_2^2
}
$$

と定義し、報酬を

$$
r_{i,t}=d_{i,t-1}-d_{i,t}
$$

とする。初期推定値は $\hat{\boldsymbol{\theta}}_{i,0}=\mathbf{0}$ とし、
推定誤差が増加した遷移では負の報酬をクリップせずに使用する。真の能力値は
シミュレーション学習時の報酬計算と評価時の報酬記録にのみ使用し、DQNの状態や
行動選択には含めない。

ネットワーク、MAP推定、replay buffer、行動マスク、TDターゲット、seed、
validationによるcheckpoint選択は通常DQN版と共通とする。モデルおよび結果の
出力名には`DQN_MAP_RMSE_reward`を含め、log-determinant増分報酬版と区別する。

#### 3.2.3. 負のMSE報酬版

log-determinant増分報酬およびRMSE減少量報酬との比較用に、通常DQNの報酬だけを
各受検者の回答後の推定誤差に基づく負のMSEへ変更したNotebookを
`EXP_v5/src/Train_and_test_DQN_MAP_negative_MSE_reward_on_the_simulated_bank.ipynb`
として実装する。受検者 $i$ のステップ $t$ における報酬を

$$
r_{i,t}
=
-\frac{1}{2}
\left\|
\hat{\boldsymbol{\theta}}_{i,t}
-
\boldsymbol{\theta}_{i,\mathrm{true}}
\right\|_2^2
$$

とする。これは2次元の推定誤差についての負の平均二乗誤差である。報酬は回答後の
MAP推定値 $\hat{\boldsymbol{\theta}}_{i,t}$ から直接計算し、直前ステップからの
誤差減少量は使用しない。真の能力値はシミュレーション学習時の報酬計算と評価時の
報酬記録にのみ使用し、DQNの状態や行動選択には含めない。

ネットワーク、MAP推定、replay buffer、行動マスク、TDターゲット、seed、
validationによるcheckpoint選択は通常DQN版と共通とする。モデルおよび結果の
出力名には`DQN_MAP_negative_MSE_reward`を含め、他の報酬版と区別する。

### 3.3. 初期段階における特異性への対処

2次元2PLでは、1項目のフィッシャー情報量行列は

$$
\mathbf{I}_j(\boldsymbol{\theta})
=
P_j(\boldsymbol{\theta})
\left(1-P_j(\boldsymbol{\theta})\right)
\mathbf{a}_j\mathbf{a}_j^\top
$$

であり、外積 $\mathbf{a}_j\mathbf{a}_j^\top$ に基づく階数1の行列である。そのため、項目情報だけを用いた場合には、開始時の情報行列はゼロ行列であり、1項目を追加した後も

$$
\det\mathbf{I}_j(\boldsymbol{\theta})=0
$$

となる。このままでは、以下の問題が生じる。

- 第1項目について、D-optimalityの行列式がすべてゼロになり、選択基準が定まらない。
- $\log\det\mathbf{I}_0$ および $\log\det\mathbf{I}_1$ が有限値にならず、DQNの初期報酬を定義できない。
- 1回答だけでは2次元能力を識別できず、無制約MLEは有限かつ一意な推定値を持たない。

本実験では、DQNとD-optimalityの両方で同じMAP推定を使用し、その事前情報行列 $\boldsymbol{\Sigma}_0^{-1}=\mathbf{I}_2$ を情報行列に加えることで対処する。これにより、$\mathbf{J}_0$ および $\mathbf{J}_1$ は正定値となり、開始時からD-optimalityの項目比較とDQNのlog-determinant報酬を有限値として計算できる。また、ガウス事前分布による正則化により、初期段階でも有限かつ一意なMAP推定値を得る。

DQNとD-optimalityの比較条件を揃えるため、事前分布、MAP推定方法、および事前情報行列の扱いは両手法で共通とする。

### 3.4. Random baseline

ランダムベースラインでは、各受検者について未出題の全項目から等確率で1項目を選択し、同一受検者への重複出題を行わない。固定seedによる1回の評価とし、項目系列を生成する乱数と回答を生成する乱数には独立した乱数生成器を使用する。

- 項目選択seed：`20260432`
- 回答生成seed：`20260430`
- 能力推定：DQNおよびD-optimalityと同じ $N(\mathbf{0},\mathbf{I}_2)$ に基づくMAP推定
- 評価対象：同じbank・能力相関条件の固定test theta
- Notebook：`EXP_v5/src/Test_Random_MAP_on_the_simulated_bank.ipynb`
- 出力：`EXP_v5/results/records_*_Random_MAP_python.csv` および `EXP_v5/results/summary_*_Random_MAP_python.csv`

## 4. 評価指標

DQN、D-optimality、ランダムベースラインの能力推定精度を比較するため、各ステップ $t$ において、各能力次元ごとのRMSEおよび2次元全体のRMSEを算出する。

ここで、$N$ は評価対象となる受検者数、$\hat{\theta}_{ik}^{(t)}$ は受検者 $i$ の第 $k$ 次元能力について、$t$ 問終了時点で得られた推定値を表す。

まず、第1次元の能力 $\theta_1$ に対するRMSEを

$$
\mathrm{RMSE}_{\theta_1}(t)
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\hat{\theta}_{i1}^{(t)}
-
\theta_{i1}
\right)^2
}
$$

とする。

同様に、第2次元の能力 $\theta_2$ に対するRMSEを

$$
\mathrm{RMSE}_{\theta_2}(t)
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\hat{\theta}_{i2}^{(t)}
-
\theta_{i2}
\right)^2
}
$$

とする。

さらに、2次元の能力推定精度をまとめて評価するため、overall RMSEを

$$
\mathrm{RMSE}_{\mathrm{overall}}(t)
=
\sqrt{
\frac{1}{2N}
\sum_{i=1}^{N}
\left[
\left(
\hat{\theta}_{i1}^{(t)}
-
\theta_{i1}
\right)^2
+
\left(
\hat{\theta}_{i2}^{(t)}
-
\theta_{i2}
\right)^2
\right]
}
$$

とする。

以上の

$$
\mathrm{RMSE}_{\theta_1}(t),\qquad
\mathrm{RMSE}_{\theta_2}(t),\qquad
\mathrm{RMSE}_{\mathrm{overall}}(t)
$$

の3つの指標を、各ステップ $t$ においてDQN、D-optimality、ランダムベースラインの各手法についてそれぞれ算出する。

これにより、各能力次元における推定精度の違いに加えて、2次元全体としての能力推定精度を比較する。また、各ステップにおけるRMSEの推移を比較することで、出題数の増加に伴ってDQNとD-optimalityの推定精度がどのように変化するかを評価する。
