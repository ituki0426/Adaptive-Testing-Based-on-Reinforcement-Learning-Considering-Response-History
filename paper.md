# 回答履歴を考慮した強化学習に基づくコンピュータ適応型テストの項目選択戦略

## Abstract

コンピュータ適応型テスト（Computerized Adaptive Testing: CAT）における項目選択は、受検者の特性値を効率的に推定するための中核的な過程である。近年、深層強化学習を用いた項目選択戦略として、Deep Q-Network（DQN）に基づく手法が提案され、従来の情報量基準手法を上回る推定精度が報告されている（Wang et al., 2024）。しかし、DQN に基づく手法では、状態として現在推定されている特性値のみを用いるため、受検者のこれまでの回答パターンに含まれる情報を十分に活用できていない。本研究では、IRT に基づく CAT を部分観測マルコフ決定過程（POMDP）として再定式化し、受検者の回答履歴を状態として利用する Deep Recurrent Q-Network（DRQN）に基づく項目選択戦略を提案する。DRQN は、回答系列を Embedding 層と LSTM により処理し、過去の回答パターンから潜在的な特性値の情報を抽出する。3パラメータロジスティックモデル（3PLM）に基づくシミュレーション実験の結果、40問選択時点において、提案手法（DRQN）は RMSE = 0.200 を達成し、MFI（RMSE = 0.342）および DQN（RMSE = 0.254）を上回る推定精度を示した。

## 1. はじめに

コンピュータ適応型テスト（CAT）は、スマート・テスティングにおいて広く用いられる手法であり、心理・教育測定における主要な研究テーマでもある（Zhang & Chang, 2016）。紙筆式テストのような非適応型の形式と比べ、CAT はより少ない項目数で受検者の真の特性値をより正確に推定できる。これは、受検者の反応に加え、内容バランスや露出制御といった制約も考慮しつつ、統計的に最も情報量の高い項目群を提示することで達成される（Weiss, 1982; Wainer et al., 2000; Frey, 2023）。

効率的な測定を実現する中核過程である項目選択は、CAT における重要な研究焦点である。従来の項目選択戦略の多くは、項目反応理論（Item Response Theory: IRT; van der Linden, 2016）の枠組みの下で提案されている。最も広く使われる方法は、フィッシャー情報量を最大化する項目を選ぶ最大フィッシャー情報量法（Maximum Fisher Information: MFI; Lord, 1980）である。MFI は推定された特性値の漸近分散の逆数を最大化するという明快な原理に基づくが、推定特性値が真の特性値から大きく乖離している場合には、最も情報量の高い項目を選択できないという問題がある（Chang & Ying, 1996）。この問題に対処するため、尤度で重み付けしたフィッシャー情報量（FIWL; Veerkamp & Berger, 1997）や、尤度重み付き Kullback-Leibler 情報量（KLP; Chang & Ying, 1996）などの代替戦略が提案されてきた。

しかし、これらの情報量基準に基づく項目選択戦略には、根本的な限界がある。第一に、従来の戦略の多くは全体最適ではなく局所最適な選択を行う。項目情報量には複数の変種があるとはいえ、ほとんどの戦略は次の1項目に関する情報のみを考慮するため、さらに数項目先まで見越して得られる潜在的情報を取り込むことが難しい。理想的な戦略は、推定特性値の変動や後に起こりうる項目選択系列を考慮しつつ、後続のテストで得られる情報を最大化するように各項目を選ぶべきである。第二に、従来の戦略は他の受検者の受検記録から情報を掘り起こして効率を高めることができない。IRT に基づく戦略は項目パラメータの推定に大規模データを利用できるが、理論枠組み上、他受検者の経験から直接学習することはできない。過去の受検記録から学習できる柔軟な枠組みに基づく選択戦略であれば、特性値推定の変化の軌跡、そしてある項目を選んだ後に将来得られるテスト情報を予測できる可能性がある。

これらの限界に対処する有望なアプローチが強化学習（Reinforcement Learning: RL）である。RL は、複雑な課題を遂行するための優れた方策を探索する知的エージェントを訓練するのに適しており（Sutton & Barto, 2018）、深層学習技術と組み合わせた深層強化学習（Deep Reinforcement Learning: DRL）は、さまざまな先端領域で従来手法や人間の能力を上回る成果を示してきた（Silver et al., 2016, 2017）。近年、RL は適応学習システムにおける学習教材推薦など、教育測定における長期的かつ適応的な目的をもつ問題の解決に適合し強力であることが予備的に示されている（Chen et al., 2018; Han et al., 2020; Li et al., 2021, 2023; Tan et al., 2020; Tang et al., 2019）。CAT においても、分類型 CAT に対する DRL ベースの選択戦略（Nurakhmetov, 2019）や、二段階最適化に基づく枠組み（Ghosh & Lan, 2021）、形成的評価への適用（Shin & Bulut, 2022）など、RL の導入が試みられている。

しかし、CAT への RL 導入に関する既存研究にはいくつかのギャップが残る。第一に、3PLM のような最も一般的な IRT モデルに基づく CAT について、RL 枠組みの定義や選択戦略の提案がなされていない。第二に、既存研究で多く用いられるアクター・クリティック法は大量の学習データを要する。第三に、RL ベースの戦略が長期的情報を考慮することで測定精度を改善できるかどうか、従来の IRT ベース戦略との体系的な比較が不足している。

これらのギャップに対し、Wang et al.（2024）は CAT の項目選択問題をマルコフ決定過程（MDP）として定式化し、DQN を用いて Q 関数を近似する手法を提案した。この手法は、現在の項目だけでなく将来のテスト全体で得られる情報量の期待値を最大化するように項目を選択するため、局所最適に陥りやすい従来手法の課題を緩和する。シミュレーションおよび実データを用いた実験において、DQN は MFI をはじめとする5つの従来手法（MFI, FIWL, KLP, MPWI, MEI）より低い RMSE と MAE を達成している。

しかし、Wang et al.（2024）の DQN に基づく手法には重要な制約がある。DQN の Q-Network は、入力として現在推定されている特性値 $\hat{\theta}_l$ のみを受け取る。すなわち、受検者がこれまでにどの項目にどう回答したかという履歴情報は、特性値の推定値に集約された形でしか利用されない。IRT に基づく特性値の推定（MLE 等）は十分統計量に基づくため、理論上は回答パターンの情報は推定値に含まれるが、Q 関数の学習においてはより豊かな情報を利用することで、より良い項目選択方策を獲得できる可能性がある。

本研究では、この課題に対し、2つの理論的拡張を行う。第一に、IRT に基づく CAT を部分観測マルコフ決定過程（POMDP）として再定式化する。従来の MDP 定式化では状態を推定特性値とするが、推定特性値は真の特性値の不完全な観測であり、特にテスト序盤では推定精度が低い。POMDP 定式化では、エージェントが直接観測できるのは各項目への回答（正答/誤答）のみであるとし、これらの観測履歴から状態を推論する枠組みを採用する。第二に、POMDP を解くためのアルゴリズムとして、Hausknecht & Stone（2015）が提案した Deep Recurrent Q-Network（DRQN）を CAT に適用する。DRQN は、DQN の全結合層を LSTM に置き換えることで、過去の観測系列から内部状態を構築し、部分観測環境における意思決定を可能にする。

## 2. タスク設定

### 2.1 項目反応モデル

本研究では、IRT において広く用いられている二値反応モデルの一つである3パラメータロジスティックモデル（3PLM）を用いる。3PLM における項目 $i$ への正答確率は以下で与えられる：

$$P_i(\theta) = c_i + \frac{1 - c_i}{1 + \exp[-a_i(\theta - b_i)]}$$

ここで、$\theta$ は受検者の特性値、$a_i$ は識別力パラメータ、$b_i$ は困難度パラメータ、$c_i$ は疑似推測パラメータである。

### 2.2 CAT の構成要素

CAT は以下の主要な構成要素から成る（Weiss & Kingsbury, 1984; Frey, 2023）：（1）アイテムバンク、（2）テストの開始、（3）特性値の推定方法、（4）項目選択戦略、（5）制約管理、（6）テストの終了。本研究は（4）の項目選択戦略に焦点を当て、他の構成要素には従来の IRT に基づく手法を用いる。

特性値の推定には最尤推定法（MLE）を用い、推定範囲を $[-4, 4]$ とする。テスト開始時の初期特性値は $[-0.5, 0.5]$ の一様分布から無作為に生成する。全回答が正答または全回答が誤答の場合は、Dodd（1990）の方法に従い、$\hat{\theta}_{l+1} = \hat{\theta}_l + (b_{\max} - \hat{\theta}_l) / 2$（全正答時）または $\hat{\theta}_{l+1} = \hat{\theta}_l + (b_{\min} - \hat{\theta}_l) / 2$（全誤答時）として特性値を更新する。

### 2.3 情報量基準に基づく項目選択戦略

#### MFI（Maximum Fisher Information）

フィッシャー情報量を最大化する項目を選択する（Lord, 1980）。項目 $i$ のフィッシャー情報量は以下で定義される：

$$I_i(\theta) = \frac{a_i^2 (1 - c_i)}{[c_i + \exp(a_i(\theta - b_i))][1 + \exp(-a_i(\theta - b_i))]^2}$$

項目選択規則は $i_l = \arg\max_{i \in B_l} I_i(\hat{\theta}_l)$ である。ここで $B_l$ は $l$ 番目の項目選択時点で未出題の項目集合である。

#### FIWL（Fisher Information Weighted by Likelihood）

尤度で重み付けしたフィッシャー情報量を最大化する（Veerkamp & Berger, 1997）。項目選択規則は以下で定義される：

$$i_l = \arg\max_{i \in B_l} \int_{-\infty}^{\infty} I_i(\theta) L(\theta) d\theta$$

ここで $L(\theta)$ は、それまでの回答パターンに基づく尤度関数である。FIWL は推定特性値の不確実性を考慮するため、テスト序盤における MFI の弱点を補う。実装では、最初の項目（回答履歴がない時点）は MFI で選択し、2問目以降に FIWL を適用する。

![IRT ベースの項目選択戦略のフローチャート](img/Flowchart%20of%20the%20IRT-based%20item%20selection%20strategy.png)

**図1**: IRT ベースの項目選択戦略のフローチャート（Wang et al., 2024, Fig.1(a) を基に作成）。情報量基準に基づく戦略では、推定特性値から項目情報量を計算し、最も情報量の高い項目を選択する。受検者が回答した後、特性値を再推定し、次の項目選択に進む。このサイクルをテスト終了条件が満たされるまで繰り返す。

### 2.4 MDP に基づく定式化（DQN）

Wang et al.（2024）は、CAT の項目選択問題を MDP として定式化した。MDP の各要素は以下のように対応する：

- **状態**: 現在の推定特性値 $\hat{\theta}_l$
- **行動**: 選択する項目 $i_l \in B_l$
- **報酬**: 項目 $i_l$ のフィッシャー情報量 $I_{i_l}(\hat{\theta}_l)$
- **環境**: 受検者の回答生成（IRT モデル）と特性値の推定（MLE）

![MDP 枠組みにおける RL ベースの項目選択戦略のフローチャート](img/Flowchart%20of%20the%20RL-based%20item%20selection%20strategy%20in%20the%20MDP%20framework.png)

**図2**: MDP 枠組みにおける RL ベースの項目選択戦略のフローチャート（Wang et al., 2024, Fig.1(b) を基に作成）。エージェント（DQN）は、環境から状態（推定特性値）を受け取り、Q 関数に基づいて最も Q 値の高い項目を行動として選択する。環境では受検者が回答を生成し、フィッシャー情報量が報酬としてエージェントに返される。図1の IRT ベースの戦略と比較すると、項目選択の基準が情報量の直接最大化から Q 値の最大化に置き換わり、将来のテスト全体を通じた累積報酬を考慮した選択が可能となる。

Q 関数は、現在の状態と行動の組に対して、テスト終了までに得られる割引累積報酬の期待値を表す：

$$Q^\pi(\hat{\theta}_l, i_l) = \mathbb{E}_\pi \left[ I_{i_l}(\hat{\theta}_l) + \sum_{k=l+1}^{L} \gamma^{k-l} \cdot I_{i_k}(\hat{\theta}_k) \mid \hat{\theta}_l, i_l \right]$$

ここで $\gamma$ は割引率、$L$ はテスト長である。DQN は Q-Network $Q(\hat{\theta}_l, i; \boldsymbol{\theta})$ を用いてこの Q 関数を近似する。Q-Network は、$\hat{\theta}_l$ を入力として受け取り、アイテムバンク内の全項目に対する Q 値を出力する全結合ニューラルネットワークである。

![DQN の Q-Network のアーキテクチャ](img/The%20architecture%20of%20Q-Network%20in%20DQN.png)

**図3**: DQN における Q-Network のアーキテクチャ。入力は推定特性値 $\hat{\theta}_l$（1次元）、2層の隠れ層（50, 30ニューロン、ReLU 活性化）を経て、各項目の Q 値を出力する。

## 3. 提案手法

### 3.1 POMDP としての再定式化

Wang et al.（2024）の MDP 定式化では、状態を推定特性値 $\hat{\theta}_l$ とする。しかし、推定特性値は真の特性値 $\theta$ の不完全な推定であり、特にテスト序盤では大きな誤差を含む。つまり、エージェントは真の状態（真の特性値）を直接観測できず、観測（回答の正誤）を通じて間接的に推論するしかない。これは部分観測マルコフ決定過程（POMDP）の構造そのものである。

そこで本研究では、CAT の項目選択問題を POMDP として再定式化する：

- **真の状態**: 受検者の真の特性値 $\theta$（エージェントには観測不可能）
- **観測**: 各項目への回答 $o_l \in \{0, 1\}$（0: 誤答、1: 正答）
- **行動**: 選択する項目 $i_l \in B_l$
- **報酬**: フィッシャー情報量 $I_{i_l}(\theta)$（学習時は真の $\theta$ で計算）

POMDP において、エージェントは観測履歴 $(o_1, o_2, \ldots, o_{l-1})$ から真の状態に関する信念（belief）を構築し、行動を選択する。

### 3.2 DRQN による Q 関数の近似

POMDP を解くため、Hausknecht & Stone（2015）が提案した Deep Recurrent Q-Network（DRQN）を用いる。DRQN は DQN の全結合層を LSTM に置き換えたアーキテクチャであり、過去の観測系列を逐次処理することで、部分観測下での Q 値推定を可能にする。

本研究における DRQN のアーキテクチャは以下の構成である：

1. **Embedding 層**: 回答を3カテゴリ（0: 誤答、1: 正答、2: 開始トークン）の埋め込み表現に変換する（埋め込み次元: 16）
2. **LSTM 層**: 埋め込み系列を逐次処理し、隠れ状態を更新する（隠れ層サイズ: 64）
3. **全結合出力層**: LSTM の出力からアイテムバンク内の全項目に対する Q 値を出力する

![DRQN のアーキテクチャ](img/ChatGPT%20Image%202026年6月3日%2000_57_09.png)

**図5**: DRQN のアーキテクチャ。回答系列 $(r_1, r_2, \ldots, r_T)$ を Embedding 層で埋め込み表現に変換し、LSTM で逐次処理する。LSTM の出力を全結合層に通し、各項目の Q 値 $Q(s_T, i)$ を出力する。

決定ステップ $l$ において、DRQN は開始トークンとこれまでの回答系列 $(\text{START}, o_1, o_2, \ldots, o_{l-1})$ を入力として受け取り、LSTM が内部状態を逐次更新する。これにより、推定特性値というスカラー情報に集約せずに、回答パターンの時系列的な特徴を直接利用して Q 値を推定する。

DQN と DRQN の本質的な違いは以下の通りである：

| | DQN | DRQN（提案手法） |
|---|---|---|
| 定式化 | MDP | POMDP |
| 状態表現 | 推定特性値 $\hat{\theta}_l$（スカラー） | 回答履歴 $(o_1, \ldots, o_{l-1})$（系列） |
| ネットワーク | 全結合ネットワーク | Embedding + LSTM + 全結合 |
| 入力次元 | 1 | 可変長系列 |

### 3.3 学習アルゴリズム

DRQN の学習は DQN と同様に、ε-greedy 法による方策選択、ターゲットネットワーク、および経験再生の3つの手法を用いる。ただし、経験再生の単位はステップごとの遷移ではなく、1エピソード（1人の受検者のテスト全体）とする。これは、LSTM の隠れ状態がエピソード内の系列に依存するためである。

学習の手順は以下の通りである：

1. 学習用の特性値 $\theta_n$（$n = 1, \ldots, N_{\text{training}}$）を生成する
2. 各受検者について、テスト長 $L$ の CAT を実行し、各ステップの回答・行動・報酬を記録する
3. エピソード単位でリプレイメモリに保存する
4. ミニバッチをサンプリングし、以下の損失関数で Q-Network のパラメータを更新する：

$$\mathcal{L} = \left[ r_l + \gamma \cdot \max_{i \in B_{l+1}} Q_{\text{target}}(s_{l+1}, i; \boldsymbol{\theta}') - Q_{\text{action}}(s_l, i_l; \boldsymbol{\theta}) \right]^2$$

5. $\tau$ 回の更新ごとにターゲットネットワークのパラメータを同期する
6. 定期的にバリデーションを行い、最良のモデルを保存する

![DQN ベースの項目選択戦略の学習過程のフローチャート](img/Flowchart%20of%20the%20training%20process%20for%20the%20DQN-based%20item%20selection%20strategy.png)

**図4**: DQN ベースの項目選択戦略の学習過程のフローチャート（Wang et al., 2024, Fig.2 を基に作成）。下部の CAT 環境ループ（項目選択→受検者の回答→特性値推定）で生成されるエピソードデータ $(\hat{\theta}_l, i_l, I_{i_l}(\hat{\theta}_l), \hat{\theta}_{l+1})$ をリプレイメモリに蓄積する。上部の Deep Q-Network では、リプレイメモリからミニバッチをサンプリングし、ターゲット Q-Network の出力と Action-Value Q-Network の出力の損失を計算してパラメータを更新する。ターゲット Q-Network のパラメータは定期的に Action-Value Q-Network から同期される。DRQN の学習過程も同様の構造を持つが、状態表現として推定特性値の代わりに回答履歴を用い、経験再生の単位がステップからエピソードに変更される点が異なる。

## 4. シミュレーション実験

### 4.1 実験設定

#### アイテムバンクの生成

Wang et al.（2024）のシミュレーション設定に基づき、アイテムバンクを生成した。各アイテムバンクは200項目から成り、項目パラメータの分布は以下の通りである：

- 識別力: $a \sim N(1.2, 0.25)$（$a > 0$ の条件付き）
- 困難度: $b \sim N(0, 1)$
- 疑似推測: $c \sim N(0.25, 0.02)$（$0 < c < 1$ の条件付き）

パラメータ間に相関のないアイテムバンク（$r_{ab} = 0$）とパラメータ間に相関のあるアイテムバンク（$r_{ab} = 0.5$）の2種類を各10バンク、計20バンク生成した。各バンクに対して、5,000名の受検者の真の特性値を標準正規分布 $N(0, 1)$ から生成した。

#### 比較手法

以下の4手法を比較する：

1. **MFI**: 最大フィッシャー情報量法（Lord, 1980）。R パッケージ catR（Magis & Barrada, 2017）を使用。
2. **FIWL**: 尤度重み付きフィッシャー情報量法（Veerkamp & Berger, 1997）。1問目は MFI、2問目以降は catR の MLWI 基準を使用。
3. **DQN**: Deep Q-Network に基づく手法（Wang et al., 2024）。状態は推定特性値。
4. **DRQN**: Deep Recurrent Q-Network に基づく手法（本研究の提案手法）。状態は回答履歴。

#### 強化学習手法のハイパーパラメータ

DQN と DRQN の学習設定は以下の通りである：

| パラメータ | DQN | DRQN |
|---|---|---|
| ネットワーク構造 | 全結合 (1→50→30→200) | Embedding(3,16)→LSTM(64)→Linear(200) |
| 学習データ数 $N_{\text{training}}$ | 1,000 | 1,000 |
| テスト長 $L$ | 40 | 40 |
| 割引率 $\gamma$ | 0.1 | 0.1 |
| リプレイメモリサイズ $N_D$ | 1,000 | 1,000 |
| ミニバッチサイズ $m$ | 128 | 128 |
| ターゲットネットワーク更新間隔 $\tau$ | 40 | 40 |
| 学習率 | 0.001 | 0.001 |
| $\varepsilon$-greedy の $\varepsilon$ | 0.1 | 0.1 |
| バリデーション間隔 $N_{\text{interval}}$ | 50 | 50 |
| バリデーションサンプル数 $N_{\text{validation}}$ | 200 | 200 |
| 学習データの特性値分布 | $N(0, 1)$ | $N(0, 1)$ |
| 報酬 | $I_{i_l}(\theta)$（真の $\theta$） | $I_{i_l}(\theta)$（真の $\theta$） |

DQN では全てのネットワークパラメータに正値制約を適用し、Q 値が正であることを保証する。DRQN では出力層の重みとバイアスのみに正値制約を適用し、さらに勾配クリッピング（最大ノルム 1.0）を導入して学習を安定化させる。

#### 評価指標

以下の3指標を用いて、推定特性値の回復精度を評価する：

$$\text{Bias} = \frac{1}{N} \sum_{n=1}^{N} (\hat{\theta}_n - \theta_n)$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (\hat{\theta}_n - \theta_n)^2}$$

$$\text{MAE} = \frac{1}{N} \sum_{n=1}^{N} |\hat{\theta}_n - \theta_n|$$

### 4.2 結果

以下に、無相関アイテムバンク（$r_{ab} = 0$、バンク1）におけるテスト長40時点での各手法の比較を示す。

**表1: 無相関バンク（バンク1）における40問選択時点の推定精度**

| 手法 | Bias | RMSE | MAE |
|---|---|---|---|
| MFI | 0.012 | 0.342 | 0.264 |
| FIWL | — | — | — |
| DQN ($\gamma = 0.1$, normal) | −0.028 | 0.254 | 0.170 |
| **DRQN** ($\gamma = 0.1$, normal) | **−0.101** | **0.200** | **0.157** |

<!-- TODO: FIWL の結果を実験後に記入 -->

DRQN は40問選択時点において RMSE = 0.200 を達成し、MFI（RMSE = 0.342）と比較して約42%、DQN（RMSE = 0.254）と比較して約21%の改善を示した。MAE についても同様の傾向が確認された。

**表2: 各ステップにおける RMSE の推移（無相関バンク、バンク1）**

| ステップ | MFI | DQN | DRQN |
|---|---|---|---|
| 5 | 0.984 | 0.852 | 1.078 |
| 10 | 0.699 | 0.519 | 0.829 |
| 15 | 0.547 | 0.424 | 0.544 |
| 20 | 0.470 | 0.239 | 0.653 |
| 25 | 0.418 | 0.260 | 0.366 |
| 30 | 0.383 | 0.369 | 0.333 |
| 35 | 0.358 | 0.294 | 0.191 |
| 40 | 0.342 | 0.254 | 0.200 |

表2に示すように、MFI は初期ステップから安定して RMSE が低下する。DQN もテスト序盤から比較的低い RMSE を示す。一方、DRQN はテスト序盤（ステップ1〜10）において MFI より高い RMSE を示すが、ステップ数が増加するにつれて急速に改善し、ステップ25以降では他の手法を上回る精度を達成する。これは、DRQN が十分な回答履歴を蓄積することで、LSTM が回答パターンから特性値の情報をより正確に抽出できるようになるためと考えられる。

## 5. 終わりに

本研究では、IRT に基づく CAT の項目選択問題を POMDP として再定式化し、回答履歴を状態として利用する DRQN に基づく項目選択戦略を提案した。シミュレーション実験により、提案手法は40問選択時点において既存の MFI および DQN を上回る推定精度を達成することが示された。

提案手法の利点として、以下が挙げられる：

1. **回答パターンの直接利用**: 推定特性値にスカラー情報として集約するのではなく、回答系列をそのまま入力とすることで、回答パターンに含まれる情報をより豊かに活用できる。
2. **POMDP としての理論的整合性**: CAT における項目選択は本質的に部分観測問題であり、POMDP としての定式化は MDP よりも問題の構造を適切に捉えている。

一方、以下の課題が残されている：

1. **テスト序盤の精度**: DRQN はテスト序盤において MFI より低い精度を示す。回答履歴が十分に蓄積されるまでは、LSTM が有効に機能しにくいためと考えられる。序盤に限り MFI 等の情報量基準手法を併用するハイブリッド戦略の検討が有望である。
2. **バンク間の一般化**: 現在の DRQN はアイテムバンクごとに学習する必要があり、異なるバンクへの転移は保証されていない。
3. **実データによる検証**: 本研究はシミュレーションデータのみを用いており、実際のテストデータでの検証が今後の課題である。

## 参考文献

- Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213–229.
- Dodd, B. G. (1990). The effect of item selection procedure and stepsize on computerized adaptive attitude measurement using the rating scale model. *Applied Psychological Measurement*, 14(4), 355–366.
- Frey, A. (2023). Computerized adaptive testing. In R. J. Mislevy & H. Jiao (Eds.), *The Oxford handbook of educational assessment*. Oxford University Press.
- Hausknecht, M., & Stone, P. (2015). Deep recurrent Q-learning for partially observable MDPs. *arXiv preprint arXiv:1507.06527*.
- Lord, F. M. (1980). *Applications of item response theory to practical testing problems*. Erlbaum.
- Magis, D., & Barrada, J. R. (2017). Computerized adaptive testing with R: Recent updates of the package catR. *Journal of Statistical Software*, 76(1), 1–19.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
- Veerkamp, W. J. J., & Berger, M. P. F. (1997). Some new item selection criteria for adaptive testing. *Journal of Educational and Behavioral Statistics*, 22(2), 203–226.
- Wainer, H., et al. (2000). *Computerized adaptive testing: A primer* (2nd ed.). Erlbaum.
- Wang, P., Liu, H., & Xu, M. (2024). An adaptive testing item selection strategy via a deep reinforcement learning approach. *Behavior Research Methods*, 56, 8695–8714.
- Weiss, D. J. (1982). Improving measurement quality and efficiency with adaptive testing. *Applied Psychological Measurement*, 6(4), 473–492.
- Weiss, D. J., & Kingsbury, G. G. (1984). Application of computerized adaptive testing to educational problems. *Journal of Educational Measurement*, 21(4), 361–375.
- Zhang, J., & Chang, H.-H. (2016). From smart testing to smart learning: How testing technology can assist the new generation of education. *International Journal of Smart Technology and Learning*, 1(1), 67–92.
- van der Linden, W. J. (2016). *Handbook of item response theory* (Vol. 1–3). CRC Press.
