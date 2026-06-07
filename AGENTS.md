実装上の疑問があれば必ず質問してください。

このファイルは Codex 向けのリポジトリ指示です。Claude Code 向けの `CLAUDE.md` と同じ研究コンテキストを前提にしつつ、Codex で作業するときの実装・検証ルールを補足します。

## プロジェクト概要

本リポジトリは、IRT（項目反応理論）の枠組みで実施される CAT（コンピュータ適応型テスト）における項目選択アルゴリズムを研究するリポジトリである。

## Codex での作業方針

- 研究上・実装上の前提が曖昧な場合は、推測で大きな変更を入れずに質問する。
- 既存の実験結果、論文メモ、ゼミ資料と矛盾する変更を避ける。
- 未追跡ファイルや既存の変更は、ユーザーの作業である可能性が高い。明示的に依頼されない限り削除・リセット・巻き戻しをしない。
- Python コードは `pyproject.toml` の Ruff / Pyright 設定に従う。
- ファイル名にスペースが含まれるスクリプトや notebook があるため、コマンド実行時はパスを必ずクォートする。
- 研究コードでは再現性を重視し、乱数 seed、データ生成条件、評価指標の変更を明示する。
- 実験条件を変えた場合は、該当する `EXP***/`、`seminar_docs/`、`paper.md`、または `tex/main.tex` の記述更新が必要か確認する。

## 研究の背景と位置づけ

### 元論文

Wang, P., Liu, H., & Xu, M. (2024). An adaptive testing item selection strategy via a deep reinforcement learning approach. *Behavior Research Methods*, 56, 8695-8714. (`papers/s13428-024-02498-x.pdf`)

- IRT に基づく CAT をマルコフ決定過程（MDP）として定式化
- 状態：現在推定されている特性値 θ̂（スカラー）
- DQN を用いて Q 関数を近似し、項目を選択
- シミュレーション・実データの双方で MFI 等5つの従来手法を上回る RMSE・MAE を達成

### MDP 定式化の限界

DQN では状態を推定特性値 θ̂ のみとするが、これには以下の問題がある：

1. マルコフ性の違反：状態 θ̂ から次の状態 θ̂' への遷移には回答履歴に基づく尤度が必要であり、現在の θ̂ だけでは次の状態が決まらない
2. 同じ推定値でも不確実性が異なる：例えば θ̂=0.5 でも、5問解いた後と20問解いた後では推定の確からしさが異なる。スカラーの推定値だけでは、この不確実性の違いを区別できない

### 本研究の提案

CAT の項目選択問題を部分観測マルコフ決定過程（POMDP）として再定式化し、回答履歴を状態として利用する。

#### POMDP の定式化（1次元 IRT の場合）

POMDP は $(S, A, O, p(s'|s,a), p(o|s',a), r)$ で表される。CAT への対応づけ：

- 真の状態 $s$：受検者の真の特性値 θ（エージェントには観測不可能）
- 行動 $a$：未出題項目の選択 $i_t \in B_t$
- 観測 $o$：回答の正誤 $o_t = u_t \in \{0, 1\}$
- 履歴：$h_t = (o_1, a_1, o_2, a_2, \ldots, o_t)$
- 観測モデル：3PLM に基づく
  $$p(o_t | \theta, i_t) = P_{i_t}(\theta)^{o_t} (1 - P_{i_t}(\theta))^{1-o_t}$$
- belief state：$b_t(\theta) = p(\theta | h_t)$（ベイズ更新による事後分布）
  $$b_t(\theta) = \frac{p(\theta) \prod_{k=1}^{t} P_{i_k}(\theta)^{o_k}(1-P_{i_k}(\theta))^{1-o_k}}{\int p(\theta') \prod_{k=1}^{t} P_{i_k}(\theta')^{o_k}(1-P_{i_k}(\theta'))^{1-o_k} d\theta'}$$
- 状態遷移：明示的に定義しない（DQN/DRQN はモデルフリー手法のため不要）
- 報酬：フィッシャー情報量 $I_{i_l}(\theta)$（学習時は真の θ で計算）

#### DRQN の採用

POMDP を解くため、DRQN（Deep Recurrent Q-Network）を採用する。Hausknecht & Stone (2015) が提案 (`papers/1507.06527v4.pdf`)。DQN の全結合層を LSTM に置き換え、観測履歴から内部状態を構築する model-free な手法。

DRQN は回答履歴のみ $(o_1, o_2, \ldots)$ を入力として受け取り、LSTM で逐次処理することで belief state を暗黙的に学習する。

## シミュレーション設定

`papers/s13428-024-02498-x.pdf` を元にした設定で、`src/generate_theta_true.py` と `src/generate_item_banks.py` を用いてデータ生成。

- IRT モデル：3PLM
- アイテムバンク：200項目（元論文は500項目）
- パラメータ分布：a ~ N(1.2, 0.25), b ~ N(0, 1), c ~ N(0.25, 0.02)
- バンクの種類：無相関（r_ab = 0）、有相関（r_ab = 0.5）、各10バンク
- 受検者：5,000名、θ ~ N(0, 1)
- テスト長：40問
- 特性値推定：MLE、範囲 [-4, 4]
- 初期特性値：Uniform(-0.5, 0.5)

## 比較手法

1. MFI：最大フィッシャー情報量法（Lord, 1980）
2. FIWL：尤度重み付きフィッシャー情報量法（Veerkamp & Berger, 1997）。1問目は MFI、2問目以降は MLWI。
3. DQN：Wang et al. (2024) の手法。状態＝推定特性値（MDP 定式化）
4. DRQN：本研究の提案手法。状態＝回答履歴（POMDP 定式化）

## 評価指標

- Bias
- RMSE
- MAE

相関係数 r は使用しない。

## 実験結果（無相関バンク1、40問時点）

| 手法 | Bias | RMSE | MAE |
|---|---:|---:|---:|
| MFI | 0.012 | 0.342 | 0.264 |
| FIWL | 未実施 | 未実施 | 未実施 |
| DQN (γ=0.1, normal) | -0.028 | 0.254 | 0.170 |
| DRQN (γ=0.1, normal) | -0.101 | 0.200 | 0.157 |

40問選択時点で、DRQN が MFI・DQN を上回る RMSE を達成。

ただし DRQN はテスト序盤（ステップ1〜10）では MFI より RMSE が高い。回答履歴が蓄積されるにつれて精度が急速に改善する。

## リポジトリ構成

- `EXP003/`：**ベースライン実験**（最新・これを参照）。MFI・FIWL・DQN・DRQN を同条件で比較し、DRQN の基本性能を確立した実験。
  - `notebook/`：Colab 用 notebook（MFI, FIWL, DRQN）
  - `result/`：実験結果 CSV
- `src/`：共有コード
  - `models.py`：DRQN のモデル定義
  - `utils.py`：IRT ユーティリティ（RESPOND, FI, MLE 等）
  - `Functions.R`：R 版の IRT 関数群
  - `generate_item_banks.py`：アイテムバンク生成
  - `generate_theta_true.py`：真の特性値生成
- `seminar_docs/`：ゼミ発表資料・進捗メモ。研究の背景整理、POMDP 定式化、実装比較、実験結果の可視化を含む。
  - `ゼミ.md`：研究テーマ全体の導入スライド。CD-CAT の基本概念、アトリビュートプロフィール、研究目的の説明に使う。
  - `0417_ゼミ.md`：CD-CAT と強化学習の初期調査メモ。参考論文、基礎概念、質問事項の整理が中心。
  - `ゼミ_2週目.md`：Wang et al. (2024) 論文の読み下し資料。IRT-CAT を MDP として定式化する元論文の要点を整理している。
  - `ゼミ3週目.md`：再現実験の初期比較資料。MFI・DQN・DRQN・DDRQN の実装比較と代替データセットでの性能確認をまとめている。
  - `ゼミ4週目.md`：belief 入力系の拡張メモ。DBQN や ARDQN の実装方針、belief の離散化表現を説明している。
  - `ゼミ5週目.md`：シミュレーション実験の中間報告。DRQN / ADRQN / DDRQN の比較と、アイテム情報入力の効果検証を含む。
  - `ゼミ6週目.md`：実データ 5-fold CV の評価メモ。RMSE・MAE・Bias・相関の fold 別可視化が中心。
  - `ゼミ7週目.md`：項目選択挙動の可視化資料。ヒートマップ、ユニーク問題数、選択軌跡で方策の振る舞いを確認する。
  - `ゼミ_改善版.md`：研究計画の再構成スライド。CD-CAT×RL の研究ギャップ、POMDP 定式化、報酬設計候補を整理している。
  - `ゼミ_ブラッシュアップ.md`：卒研発表向けのブラッシュアップ版。背景説明と研究意義をより平易にまとめた資料。
  - `ゼミ3週目.pdf` / `ゼミ4週目.pdf` / `ゼミ5週目.pdf` / `ゼミ6週目.pdf`：対応する Markdown / Marp 資料の出力版。
- `papers/`：参考論文 PDF・Markdown 版。対応する Markdown 版（`.md`）がある論文は、内容把握や引用候補の確認では Markdown を優先し、必要に応じて PDF と照合する。
  - `s13428-024-02498-x.md`：Wang et al. (2024), *An adaptive testing item selection strategy via a deep reinforcement learning approach*。本研究の直接の出発点であり、IRT-CAT を MDP として定式化した DQN ベース項目選択法の元論文。
  - `1507.06527v4.md`：Hausknecht & Stone (2015), *Deep Recurrent Q-Learning for Partially Observable MDPs*。DRQN の原典であり、POMDP に対して LSTM を組み込んだ DQN の基本設計を確認するための論文。
  - `1704.07978v6.md`：*On Improving Deep Reinforcement Learning for POMDPs*。行動と観測を組み合わせる ADRQN 系の拡張を扱い、回答履歴に項目情報を組み合わせる設計の参考になる。
  - `363_report.md`：Maxim Egorov (2015), *Deep Reinforcement Learning with POMDPs*。belief ベース入力と history ベース入力の両方を比較する短報で、POMDP と DQN の接続を概観するのに向く。
  - `IPSJ-GPWS2018034.md`：Oh Hyunwoo, 金子知適 (2018), *LSTM の初期状態の学習による DRQN の改善*。DRQN の更新開始時 hidden state を学習的に与える改善案で、長期依存学習の弱点への対処を扱う。
  - `s11336-009-9123-2.md`：*When Cognitive Diagnosis Meets Computerized Adaptive Testing: CD-CAT*。CD-CAT の代表的基礎論文で、KL・エントロピー系の項目選択基準と CDM ベース CAT の枠組みを押さえるために使う。
  - `10.1177_0013164418790634.md`：Lin & Chang (2019), *Item Selection Criteria With Practical Constraints in Cognitive Diagnostic Computerized Adaptive Testing*。属性バランスや露出制御を含む実運用上の制約付き CD-CAT 項目選択を扱う。
  - `reinforcement-learning-applied-to-adaptive-classification-5badyfmq0l.md`：Nurakhmetov, *Reinforcement Learning Applied to Adaptive Classification Testing*。CD-CAT に近い適応的分類テストへ RL を導入する枠組みを扱い、RL による逐次出題の先行事例として参照する。
- `img/`：図
  - `Flowchart of the IRT-based item selection strategy.png`：元論文 Fig.1(a): IRT ベースの項目選択戦略のフローチャート
  - `Flowchart of the RL-based item selection strategy in the MDP framework.png`：元論文 Fig.1(b): MDP 枠組みにおける RL ベースの項目選択戦略のフローチャート
  - `Flowchart of the training process for the DQN-based item selection strategy.png`：元論文 Fig.2: DQN ベースの項目選択戦略の学習過程のフローチャート
  - `The architecture of Q-Network in DQN.png`：DQN の Q-Network アーキテクチャ
  - `The architecture of Q-Network in DRQN.png`：DRQN の Q-Network アーキテクチャ
- `data/`：生成されたアイテムバンク・特性値データ
- `EXP001/`：**Learnable Initial Hidden State DRQN**。LSTM の初期 hidden/cell state をゼロ固定から学習可能パラメータ（`nn.Parameter`）に変更し、序盤（step 1〜10）の RMSE 悪化を緩和できるか検証する実験。変更点はモデルの初期状態のみで、入力・報酬・評価手順はベースライン（EXP003）と同一。詳細は `EXP001/exp_summary.md` 参照。
- `EXP002/`：**Burn-in DRQN**。replay memory から取り出した系列の先頭 `burn_in_steps`（デフォルト5）ステップを no-grad で LSTM に流して hidden state を構築し、残りの後半区間のみで TD 損失を計算する学習法を検証する実験。モデル構造はベースライン（EXP003）と同一で、変更点は学習ループのみ。詳細は `EXP002/exp_summary.md` 参照。
- `paper.md`：研究論文の Markdown 版
- `tex/`：研究論文の LaTeX 版
  - `main.tex`：`paper.md` の LaTeX 版

## 論文 Markdown の扱い

- 論文の Markdown 化を行う場合は、原則として `papers/<PDFファイル名>.md` という対応関係を保つ。
- 図を切り出す場合は `papers/img/<PDFファイル名>_figN.png` の命名を用いる。
- 既存の Markdown 論文を更新する場合は、本文だけでなく図リンク切れや節見出しの崩れも確認する。

## 実行・検証コマンド

Python の依存関係は `uv` と `pyproject.toml` で管理する。

```bash
uv sync
uv sync --group dev
```

Python コードを変更した場合は、可能な範囲で以下を実行する。

```bash
uv run ruff check -- src
uv run ruff format --check -- src
uv run pyright
```

特定ファイルだけ確認する場合は、スペースを含むパスに注意して以下のように実行する。

```bash
uv run ruff check -- "src/Train and test DQN on the real responses.py"
uv run ruff format --check -- "src/Train and test DQN on the real responses.py"
```

R スクリプトは必要に応じて以下の形式で実行する。

```bash
Rscript "src/IRT-based CAT.R"
Rscript "src/Functions.R"
```
