---
marp: true
theme: default
paginate: true
math: mathjax
size: 16:9
style: |
  section {
    font-size: 26px;
  }
  section.lead h1 {
    font-size: 1.6em;
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    column-gap: 1rem;
    align-items: start;
    height: 100%;
  }
  .cols.single-left {
    grid-template-columns: 48% 52%;
  }
  .cols.single-right {
    grid-template-columns: 52% 48%;
  }
  .cols img {
    max-width: 100%;
    max-height: 380px;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
    margin: 0 auto;
  }
  .cols .img-block {
    text-align: center;
  }
  .cols .text-block {
    font-size: 0.82em;
    line-height: 1.35;
  }
  .cols .text-block ul {
    margin: 0.3em 0;
    padding-left: 1.1em;
  }
  section.compact {
    font-size: 22px;
  }
  section.compact h2 {
    font-size: 1.15em;
  }
  .footnote {
    font-size: 0.65em;
    margin-top: 0.4em;
    color: #444;
  }
---

<!-- _class: lead -->

# 0.参考文献

---

# 0.参考文献

[Cheng, Y. (2009). When cognitive diagnosis meets computerized adaptive testing: CD-CAT.Psychometrika, 74(4), 619-632.](https://link.springer.com/article/10.1007/s11336-009-9123-2)

[山口一大 (2016). 認知診断モデルにおける Q 行列の誤設定が診断精度に与える影響.2016 年度 日本テスト学会誌 Vol.13](https://www.jstage.jst.go.jp/article/jart/13/1/13_17/_pdf/-char/ja)

[山口一大(2022). 認知診断モデルの基本の基. 日本テスト学会 第15回記念講演ワークショップ, p. 4.](https://www.jartest.jp/pdf/15-3yamaguchi.pdf)

---

<!-- _class: lead -->

# 1.認知診断モデル(CDM)

---

## 1.1.認知診断テスト(CDA)とは？

診断テストの中でも，とくに，測定する知識構造や認知的スキルをあらかじめ考慮に入れて設計されたテストを認知診断テスト（Cognitive Diagnostic Assessment,CDA; Leighton & Gierl, 2007）という。

---

## 1.2.CDMとは？

CDM：CDAのための統計モデルを総称して，認知診断モデル（Cognitive Diagnostic Models, CDM; Leighton & Gierl,2007）と呼ぶ。

- 複数の認知要素を仮定して学習者の到達度を調べるための教育測定モデル
- 認知要素：アトリビュート
- Diagnostic classification models (DCMs)とも呼ばれる
- CDMの主目的：学習者のアトリビュートプロフィールを推定すること
- アトリビュートプロフィール：アトリビュートの習得・未習得の組み合わせ。アトリビュート習得パタン，アトリビュートパタンなどとも呼ばれる

---

## 1.2.CDMとは？（続き）

- 個人 $i$ のアトリビュート $k\ (=1,\ldots,K)$ の習得の有無を $\alpha_{ik} \in \{0,1\}$ で表す。1 はアトリビュート $k$ の習得に，0 は未習得に対応する。
- ただし、$i\ (=1,\ldots,I)$ は個人，$j\ (=1,\ldots,J)$ は項目番号を意味する。
- サイズ $I \times K$ のアトリビュート習得パタン行列を $\mathbf{A}=[\boldsymbol{\alpha}_1,\cdots,\boldsymbol{\alpha}_i,\cdots,\boldsymbol{\alpha}_I]^{t}$ で表したりする。

---

## 1.2.CDMとは？ — 図と説明

<div class="cols single-left">
<div class="img-block">

![アトリビュートプロフィールの例](img/分数計算におけるアトリビュート習得パタン.png)

</div>
<div class="text-block">

**図の出典**：山口一大・岡田謙介（2017）「近年の認知診断モデルの展開」『行動計量学』44巻2号, 181–198, 表2「分数の計算におけるアトリビュート習得パタン」。

- 例えば項目「1/3 + 4/3」に対しては，「加減」のアトリビュートを習得している習得パタン（2, 5, 6, 8）の個人は正答する可能性が高く，「加減」アトリビュートを習得していないパタン（1, 3,4, 7）の個人は正答する可能性が低いと考えられる。(最も基本的なCDMであるDINAモデルの考え方)
- CDMはカテゴリカルな潜在説明変数（アトリビュート）と Q 行列を用いて項目反応確率を規定すること，アトリビュート習得パタンごとに項目への反応確率が異なると考える = Q行列によって，アトリビュート習得パタンごとの反応確率を定義できる

→テストの合計点のみではわからない個々人の能力の習得・未習得の状態，すなわち，個人のつまずきを診断する統計モデル

</div>
</div>

---

## 1.3.アトリビュートプロフィールを推定するため必要な要素

1. アトリビュートの集合
   - テストの認知理論や文献研究に基づいて決定される

2. Q-matrix (Tatsuoka, 1983)
   - アトリビュートと項目の関連を示した行列
   - CDMでは所与のものとされる
   - $Q$ 行列の要素 $q_{jk} \in \{0,1\}$ は，1 が項目 $j$ にアトリビュート $k$ が求められることに，0 がそうでないことに対応する。項目 $j$ についての要素をまとめて  $\boldsymbol{q}_j=[q_{j1},\cdots,q_{jk},\cdots,q_{jK}]^{t}$ とし，サイズ $J \times K$ である $Q$ 行列を $\mathbf{Q}=[\boldsymbol{q}_1,\cdots,\boldsymbol{q}_j,\cdots,\boldsymbol{q}_J]^{t}$ で表す。
   - ただし、J は項目数、K はアトリビュート数である。

---

## 1.3.— Q行列の表現

<div class="cols single-left">
<div class="img-block">

![Q行列の表現](img/Q行列の表現.png)

</div>
<div class="text-block">

**図1. Q行列の表現**

出典：山口一大（2022）「認知診断モデルの基本の基」日本テスト学会 第15回記念講演ワークショップ, p. 4.

</div>
</div>

---

## 1.3.— 項目反応関数（IRF）

3. 項目反応関数（item response function, IRF）
   - 測定モデル (measurement model)
   - アトリビュートの認知理論に依存して決定される

ex)DINAモデルの項目反応関数は以下のように定義される。

$$
\Pr(x_{ij}=1)=(1-s_j)^{\eta_{ij}}g_j^{\,1-\eta_{ij}}
$$

---

## 1.3.— 構造モデル・解答データ

4. 構造モデル (structural model)
   - アトリビュートの間の関係のモデル

5. 解答データ(項目反応データ)
   - 実際の項目反応のデータ
   - $x_{ij} \in \{0,1\}$ は個人 $i$ の項目 $j$ への反応であり，1 は正答，0 は誤答に対応する。 項目反応をまとめたサイズ $I \times J$ の行列を $\mathbf{X}$ とする。
   - ただし、$i\ (=1,\ldots,I)$ は個人，$j\ (=1,\ldots,J)$ は項目番号を意味する。

---

## 1.3.— 項目反応行列の表現

<div class="cols single-left">
<div class="img-block">

![項目反応行列の表現](img/項目反応行列の表現.png)

</div>
<div class="text-block">

**図2. 項目反応行列の表現**

出典：山口一大（2022）「認知診断モデルの基本の基」日本テスト学会 第15回記念講演ワークショップ, p. 4.

</div>
</div>

---

## 1.3.— 認知診断モデル（CDM）の概念図

<div class="cols single-right">
<div class="text-block">

**図3. 認知診断モデルの模式図**

出典：山口一大（2022）「認知診断モデルの基本の基」日本テスト学会 第15回記念講演ワークショップ, p. 4.

</div>
<div class="img-block">

![認知診断モデル（CDM）の概念図](img/認知診断モデル（CDM）の概念図.png)

</div>
</div>

---

# 補償モデルと非補償モデル

- 非補償モデルは，項目に関連する複数のアトリビュートのうち，全てが正答のためには必須であり，1 つでも求められるアトリビュートを習得していない場合には正答確率が大きく低下することが特徴。
- 補償モデルの特徴は，正答のために求められるアトリビュートのうち未習得のものがあったとしても，ほかに習得しているアトリビュートがあれば，それに応じて正答確率が上がることである。

---

## 認知診断モデルの例

### 1.ルールスペースモデル（Tatsuoka, 1983）

### 2.二値スキルモデル（Haertel, 1984; Haertel & Wiley, 1993）

### 3.ベイズ推論ネットワークモデル（Mislevy, Almond, Yan, & Steinberg, 1999）

### 4.DINAモデル（“Deterministic Input, Noisy ‘And’ Gate”）（Haertel, 1989; Junker & Sijtsma, 2001）

---

### 4.DINAモデル（続き）

- 基本的な発想：ある項目に正答するためには，その項目に必要な能力をすべて習得していないといけないだろう。
- 一つでもアトリビュートを習得していない場合には，正答確率は低いはず。
- １つ足りないのも全部足りないのも同じくらい低くなるのではないか。
- 非補償モデル

### 5.DINOモデル（“Deterministic Input, Noisy ‘Or’ Gate”）（Templin & Henson, 2006）

- DINAモデルに対する補償モデルが DINOモデル

### 6.Fusionモデル（Hartz, 2002; Hartz, Roussos, & Stout, 2002）

---

<!-- _class: compact -->

## 統計モデルとしてのCDM

CDM＝制約付き潜在クラスモデルだと考えることができる (Rupp, Templin, & Henson, 2010)

- CDMは潜在的でカテゴリカルな説明変数＝アトリビュートを持つ

IRT がテスト解答者を評価し，その能力特性値を与えることを主要な目的として用いられるのに対し，CDM はテスト解答者の領域ごとのスキルを診断することを主要な目的として用いられる．

CDM ではテストの問題を解くために複数のスキルが必要であると考え，各スキルを習得しているかどうかによって問題に正答できる確率が変化すると考える．このような設定において，それぞれのテスト解答者について，スキル習得有無のパタンを解答データから推定することが CDM の基本的な枠組みである．

---

<!-- _class: compact -->

## 統計モデルとしてのCDM（続き）

Rupp & Templin (2008b, p.226) は，より形式的な表現を用いて CDM を以下のように定義している（著者訳）．

CDM は単純または複雑な負荷量構造を持つ確率的，確証的な多次元潜在変数モデルである．CDMはカテゴリカルな観測従属変数のモデリングに適しており，潜在的でカテゴリカルな説明変数を持つ．説明変数は補償的・非補償的に統合されて潜在クラスを生成する」

このように CDM は，通常「正答・誤答」に対応する従属変数のみならず，「あるスキルの習得・非習得」のように説明変数にもカテゴリカル変数を仮定するモデルであって，潜在クラス分析（Latent ClassAnalysis, LCA; Lazarsfeld & Henry, 1968）モデルの特殊な場合とみることができる

---

<!-- _class: compact -->

## 統計モデルとしてのCDM（続き2）

すなわち，カテゴリカルで潜在的な説明変数から潜在クラスが定義され，この潜在クラスごとに観測従属変数への反応確率が異なると考えるのである．

先にあげた，潜在的でカテゴリカルな説明変数はアトリビュート（attribute）と呼ばれる．

アトリビュートは問題項目に正答するために求められる領域ごとの認知能力やスキルの要素に対応する．このアトリビュートに関連した重要な要素として，Q 行列（Tatsuoka,1983）とアトリビュート習得パタン行列がある．

因子分析との対応でいえば，アトリビュート，Q 行列，アトリビュート習得パタンは，それぞれ因子，因子負荷量行列，因子得点に相当する．

---

# コンピュータ適応型テスト（CAT）

- コンピュータ適応型テスト(Computerized Adaptive Testing)とは、各受験者にとって「最も適合する」項目を見つけようとするテスト方式 → 結果、従来の紙筆方式と比べて、テストの効率と精度を向上させることができる。
- ex)正解が続けば、より難しい問題を出題、不正解が続けば、少し簡単な問題を出題
- CATプログラムの鍵となるのは、最適な項目を逐次的に選択する**項目選択アルゴリズム**
- 最もよく知られた項目選択法の 1 つが、**最大フィッシャー情報量（MFI）法**（Lord, 1980; Thissen & Mislevy, 2000）。

---

## CAT — フィッシャー情報量

- フィッシャー情報量は、観測可能な確率変数 $X$ が、未知の母数 $\theta$ についてどれだけの情報を持っているかを測るものであり、この $\theta$ に対して尤度関数が依存している。数式で表すと、次のようになる。

$$
I(\theta)
= E \left\{ \left[ \frac{\partial}{\partial \theta} \ln f(X; \theta) \right]^2 \middle| \theta \right\}
= E \left\{ \left[ \frac{\partial}{\partial \theta} \log_e f(X; \theta) \right]^2 \middle| \theta \right\}.
$$

---

## CAT — MFI とその他

- $f(X;\theta)$ は項目反応関数（たとえば、1PL、2PL、3PL モデル；Hambleton & Swaminathan, 1985 を参照）に基づいて計算される尤度関数を表し、$\theta$ は関心のある潜在特性である。$\theta$ を能力（ability）と呼ぶ
- いま、$t$ 個の項目がすでに実施されたとする。MFI アルゴリズムは、$\hat{\theta}_i^{(t)}$、すなわち最新の能力推定値において、フィッシャー情報量が最大となる次の項目を選択する。言い換えると、受験者 $i$ に対する第 $(t+1)$ 番目の項目は次のように与えられる。

$$
\arg \max_{h} \left\{ I_h \left( \hat{\theta}_i^{(t)} \right) : h \in R^{(t)} \right\}
$$

---

## CAT — MFI（続き）と CD-CAT への動機

- ここで、$R^{(t)}$ は段階 $t$ において利用可能な項目集合を表す。能力推定値、すなわち $\hat{\theta}$ の分散はフィッシャー情報量に反比例するため、中間的な能力推定値においてフィッシャー情報量を最大化する項目を選択することで、MFI アルゴリズムは真の $\theta$ を素早く特定することができる。
- **Kullback–Leibler 情報量**および**Shannon エントロピー**に基づくアルゴリズムも使われる
- MFI アルゴリズムは偶然による当たり外れに影響されやすい。したがって、CAT の初期段階では、より大域的な尺度の方が有用である可能性がある。
- その一例が、Kullback–Leibler 情報量に基づいて開発された Chang and Ying（1996）の**大域的情報量法**(Global Information Approach)
- 認知診断においても、コンピュータ適応型テストを用いることで、より効率的に認知診断を行える可能性があるのではないか？ = 「認知診断型コンピュータ適応テスト（CD-CAT）」

---

# 認知診断型適応テスト(CD-CAT)

- Xu, Chang, and Douglas（2003）は、認知診断型コンピュータ適応テスト（CD-CAT）のための 2 つの項目選択ヒューリスティックを具体的に検討した。1 つは Kullback–Leibler 情報量に基づくもので、もう1つは Shannon エントロピーに基づくもの

---
