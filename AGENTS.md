# AGENTS.md

実装上・研究上の疑問があれば必ず質問してください。

このファイルは Codex 向けのリポジトリ指示です。現在の `refactoring.md` を研究コンテキストの根拠とし、Claude Code 向けの `CLAUDE.md` と同じ前提を共有します。

## プロジェクト概要

本リポジトリは、IRT（項目反応理論）の枠組みで実施される CAT（コンピュータ適応型テスト）における項目選択アルゴリズムを研究するリポジトリです。

現在の主目的は、Wang et al. (2024) の DQN(normal) と MFI のシミュレーション実験を再現することです。

## Codex での作業方針

- 研究上・実装上の前提が曖昧な場合は、推測で大きな変更を入れずに質問してください。
- 既存の実験結果、論文メモ、ゼミ資料、`refactoring.md`、`CLAUDE.md` と矛盾する変更を避けてください。
- 未追跡ファイルや既存の変更は、ユーザーの作業である可能性が高いです。明示的に依頼されない限り削除・リセット・巻き戻しをしないでください。
- Python コードは `pyproject.toml` の Ruff / Pyright 設定に従ってください。
- ファイル名にスペースが含まれるスクリプトや notebook があるため、コマンド実行時はパスを必ずクォートしてください。
- 研究コードでは再現性を重視し、乱数 seed、データ生成条件、評価指標、報酬、状態、行動マスク、評価対象を変更した場合は明示してください。
- 実験条件を変えた場合は、該当する `EXP***/exp_summary.md`、`refactoring.md`、`CLAUDE.md`、`seminar_docs/`、`paper.md`、または `tex_ver*/main.tex` の更新が必要か確認してください。
- DQN の RMSE だけを見て判断せず、必ず同じデータセット上の MFI と比較してください。

## 最重要の現状認識

現時点では、元論文 Table 1 の「DQN(normal) が MFI より低い RMSE を示す」という主要結果は再現できていません。

特に、MFI が元論文に近い `data/3PL/`（sqrt あり）条件では、DQN が MFI に負けています。この状態を前提として、DQN が良いはずという方向へ説明やコードを合わせないでください。

## 元論文

Wang, P., Liu, H., & Xu, M. (2024). An adaptive testing item selection strategy via a deep reinforcement learning approach. *Behavior Research Methods*, 56, 8695-8714. (`papers/s13428-024-02498-x.pdf`)

元論文 Table 1（非相関バンク、RMSE）：

| Test length | DQN (normal) | MFI |
|---:|---:|---:|
| 10 | 0.415 | 0.493 |
| 20 | 0.278 | 0.331 |
| 30 | 0.244 | 0.277 |
| 40 | 0.227 | 0.245 |

## データセット

本リポジトリでは、非相関アイテムバンクとして 2 種類のデータセットを使っています。

| データセット | 項目数 | 分散指定の扱い | 主な使用実験 | 位置づけ |
|---|---:|---|---|---|
| `data/uncorrelated_banks/` | 500 | 分散値に `sqrt` を取らずに生成 | EXP007, EXP008, EXP011 | 初期のローカル再現実験で使用したバンク |
| `data/3PL/` | 500 | 分散値に `sqrt` を取り、標準偏差として生成 | EXP009, EXP010 | 元論文 Table 1 の再現により近いバンク |

`data/3PL/` は `src/generate_item_banks_3pl.R` によって生成された `data/3PL/item_bank_uncor_1.csv` から `data/3PL/item_bank_uncor_10.csv` を使用します。元論文のパラメータ設定では a, b, c のばらつきは分散として記述されているため、乱数生成時には `sqrt(分散)` を標準偏差として渡す必要があります。

`data/uncorrelated_banks/` はこの `sqrt` 処理を行っていないため、項目パラメータの分布が `data/3PL/` と異なります。

## 結果保存先

MFI の結果は、評価対象のデータセットごとに保存先が異なります。

| MFI の対象データセット | 分散指定の扱い | MFI 結果保存先 |
|---|---|---|
| `data/uncorrelated_banks/` | 分散値に `sqrt` を取らずに生成 | `EXP007/results/` |
| `data/3PL/` | 分散値に `sqrt` を取り、標準偏差として生成 | `data/3PL/results/` |

DQN の各実験結果は `EXP(実験番号)/results/` 内に保存します。

## 元論文コードの問題点

元論文の DQN プログラムは `Train and test DQN on the simulated banks.py` です。

このコードには、再現実験へ影響しうる以下の問題があります。

- パスが `E:/...` の Windows パスにハードコードされており、そのままでは動かない。
- `RESPOND` の乱数生成が `np.random.rand(1)` で、全項目に同じ乱数が使われる。
- `Choose_Action` の ε-greedy 判定に `np.random.randn()` が使われている。探索確率の判定には `np.random.rand()` が妥当。
- `action_space = 500` がハードコードされており、バンクサイズと一致しない場合がある。

EXP007 以降でこれらを修正しています。

## 実験一覧

### EXP007

DQN ベースライン（元論文再現・ローカル動作版）。

Notebook：

`EXP007/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

使用バンク：

`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）

DQN 結果保存先：

`EXP007/results/`

詳細：

`EXP007/exp_summary.md`

### EXP008

TD ターゲット計算の修正。

Notebook：

`EXP008/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

EXP007 の replay buffer 学習で Q(next) を計算する際に、出題済み項目が候補に含まれてしまう問題を修正しました。

主な変更点：

- `RESPOND` の乱数生成を項目ごとに独立した乱数へ修正。
- `MLE` / `MLE_TEST` に `np.clip(p, 1e-10, 1 - 1e-10)` を追加し、`log(0)` を防止。
- memory に `terminal` フラグと `next_available` マスクを追加。
- Q(next) を `masked_fill(-inf)` で出題済み項目を除外してから argmax。
- 終端ステップでは Q_next をゼロにして TD ターゲットを計算。

使用バンク：

`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）

DQN 結果保存先：

`EXP008/results/`

主要結果（bank 1、RMSE）：

| gamma | step 10 | step 20 | step 30 | step 40 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.636 | 0.449 | 0.376 | 0.341 |
| 0.3 | 0.662 | 0.465 | 0.385 | 0.351 |

TD ターゲットのマスク修正だけでは DQN の明確な改善は確認できませんでした。

### EXP009

`data/3PL/` バンクへの切り替え。

Notebook：

`EXP009/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 のアルゴリズムを維持しつつ、バンクを `data/3PL/`（500項目、分散値に `sqrt` を取って生成）に変更した実験です。

DQN 結果保存先：

`EXP009/results/`

詳細：

`EXP009/exp_summary.md`

### EXP010

RESPOND 修正 + TD ターゲット修正 + `data/3PL/`。

Notebook：

`EXP010/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 に対して RESPOND バグ修正、TD ターゲット修正、`data/3PL/` への切り替えを同時に適用した実験です。

| 変更内容 | EXP007 | EXP008 | EXP009 | EXP010 |
|---|---|---|---|---|
| RESPOND バグ修正 | yes | yes | yes | yes |
| TD ターゲットのマスク修正 | no | yes | no | yes |
| data/3PL（sqrt あり）使用 | no | no | yes | yes |

使用バンク：

`data/3PL/`（500項目、分散値に `sqrt` を取って生成）

DQN 結果保存先：

`EXP010/results/`

比較用プロット：

`EXP010/notebook/plot_rmse_comparison.ipynb`

詳細：

`EXP010/exp_summary.md`

### EXP011

DQN with Estimated theta-hat Reward（論文アルゴリズム準拠）。

Notebook：

`EXP011/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

EXP007 との差は、報酬の計算に使う theta の違いのみです。EXP007 から EXP010 では元論文コードに合わせて真の theta で Fisher 情報量を計算していましたが、元論文 Algorithm 1 と Eq. 13 では推定値 theta-hat で計算した Fisher 情報量が報酬として定義されています。

使用バンク：

`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）

DQN 結果保存先：

`EXP011/results/`

主要結果（bank 1、RMSE）：

| gamma | step 10 | step 20 | step 30 | step 40 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.626 | 0.442 | 0.370 | 0.330 |
| 0.3 | 0.700 | 0.481 | 0.398 | 0.359 |
| 0.9 | 0.934 | 0.694 | 0.553 | 0.482 |

推定 theta-hat で報酬を計算しても、EXP007 に対する明確な改善は確認できませんでした。

## 現状の再現性評価

### 結論

現時点では、元論文 Table 1 の「DQN(normal) が MFI より低い RMSE を示す」という主要結果は再現できていません。

データセットによって再現できている部分が異なります。

- `data/uncorrelated_banks/`（sqrt なし）では、DQN が MFI よりやや低い RMSE になる場合があります。しかし MFI 自体が元論文より大幅に悪く、この条件は元論文 Table 1 のデータ設定を再現できていません。
- `data/3PL/`（sqrt あり）では、MFI は元論文にかなり近づきます。しかしこの条件では、DQN は step 20 から 40 で MFI より RMSE が高く、元論文の DQN 優位性を再現できていません。

現状は「MFI が元論文に近い条件では DQN が MFI に負ける」状態であり、元論文の結論は未再現です。

### MFI の絶対値再現

| Test length | 元論文 MFI | EXP007 MFI（uncor, sqrt なし） | EXP009/010 MFI（3PL, sqrt あり） |
|---:|---:|---:|---:|
| 10 | 0.493 | 0.701 | 0.504-0.522 |
| 20 | 0.331 | 0.462 | 0.327-0.340 |
| 30 | 0.277 | 0.377 | 0.271-0.278 |
| 40 | 0.245 | 0.336 | 0.240-0.245 |

`data/3PL/`（sqrt あり）は MFI の絶対値再現にはかなり近いです。一方、`data/uncorrelated_banks/`（sqrt なし）は MFI が元論文より大幅に悪いです。

### DQN と MFI の比較

元論文の主張は、同じテスト長で DQN(normal) の RMSE が MFI より低いことです。再現実験でも、DQN と MFI は同じデータセット上で比較してください。

sqrt なしデータセット（`data/uncorrelated_banks/`）：

| Test length | MFI | EXP007 DQN gamma=0.1 | DQN の改善量（MFI - DQN） | 元論文の改善量 | 優劣（RMSE） |
|---:|---:|---:|---:|---:|---|
| 10 | 0.701 | 0.653 | 0.048 | 0.078 | DQN < MFI |
| 20 | 0.462 | 0.443 | 0.019 | 0.053 | DQN < MFI |
| 30 | 0.377 | 0.367 | 0.010 | 0.033 | DQN < MFI |
| 40 | 0.336 | 0.329 | 0.007 | 0.018 | DQN < MFI |

sqrt なしでは DQN が MFI より低い RMSE を示しますが、改善量は元論文 Table 1 よりかなり小さいです。さらに MFI の RMSE 自体も元論文から大きく外れているため、この結果だけでは元論文 Table 1 の再現とは言えません。

sqrt ありデータセット（`data/3PL/`）：

| Test length | 元論文 DQN | 元論文 MFI | local MFI | EXP010 DQN best | 優劣（RMSE） |
|---:|---:|---:|---:|---:|---|
| 10 | 0.415 | 0.493 | 0.504 | 0.470 | DQN < MFI |
| 20 | 0.278 | 0.331 | 0.340 | 0.359 | DQN > MFI |
| 30 | 0.244 | 0.277 | 0.278 | 0.303 | DQN > MFI |
| 40 | 0.227 | 0.245 | 0.245 | 0.277 | DQN > MFI |

`EXP010 DQN best` は、bank 1 で実施した gamma=0.05, 0.1, 0.3, 0.5, 0.7 のうち、各 step で最も低い RMSE を示した値です。最も有利な値を選んでも、step 20 から 40 では DQN が MFI より悪いです。

特に step 40 では、元論文は DQN 0.227 < MFI 0.245 ですが、再現実験では DQN 0.277 > MFI 0.245 となっています。

## 現状の問題点

- EXP007 では DQN が MFI よりやや低い RMSE を示すが、MFI 自体が元論文より大幅に悪い。
- EXP007 の DQN 改善量は元論文 Table 1 よりかなり小さい。
- EXP008 の TD ターゲット修正では DQN の明確な改善は確認できなかった。
- EXP009/EXP010 では MFI は元論文に近づいたが、DQN は step 20 から 40 で MFI より悪い。
- EXP011 で報酬を論文定義どおり推定 theta-hat にしても、明確な改善は確認できなかった。
- 元論文コードと論文アルゴリズムの報酬定義の乖離だけでは、元論文 Table 1 との差は説明できない。

## 今後の確認事項

DQN が MFI を安定して上回るという元論文 Table 1 の結果は、現時点では再現できていません。そのため、追加実験の前に以下を確認してください。

- 元論文本文のアルゴリズム、数式、実験条件と、公開されている実装プログラムの間に乖離がないか確認する。
- 公開実装プログラムに、再現結果へ影響するバグが残っていないか確認する。
- 乖離やバグを見つけた場合は、それが「論文本文準拠の修正」なのか「公開コード準拠の再現」なのかを区別して記録する。
- 修正後は、DQN の RMSE だけでなく MFI との優劣を step 10, 20, 30, 40 で比較する。

## 実行コマンド

Python 依存関係：

```bash
uv sync
```

リント・型チェック：

```bash
uv run ruff check -- src
uv run ruff format --check -- src
uv run pyright
```

R スクリプト：

```bash
Rscript "src/generate_item_banks_3pl.R"
Rscript "src/Test MFI on the 3PL bank.R"
```

ファイル名にスペースが含まれるスクリプトや notebook があるため、コマンド実行時はパスを必ずクォートしてください。
