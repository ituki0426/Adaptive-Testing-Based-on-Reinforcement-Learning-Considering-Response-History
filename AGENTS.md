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
- 実験条件を変えた場合は、該当する `EXP***/`、`seminar_docs/`、`paper.md`、`refactoring.md`、または `tex_ver*/main.tex` の記述更新が必要か確認する。

## 研究の背景と位置づけ

### 元論文

Wang, P., Liu, H., & Xu, M. (2024). An adaptive testing item selection strategy via a deep reinforcement learning approach. *Behavior Research Methods*, 56, 8695–8714. (`papers/s13428-024-02498-x.pdf`)

IRT に基づく CAT を MDP として定式化し、DQN で項目選択を学習する手法を提案した論文。シミュレーション実験では MFI 等の従来手法を上回る RMSE を達成している。

### 本ディレクトリの目的

元論文における DQN(normal) と MFI のシミュレーション実験を再現すること。

元論文 Table 1（非相関バンク、RMSE）：

| Test length | DQN (normal) | MFI |
|---:|---:|---:|
| 10 | 0.415 | 0.493 |
| 20 | 0.278 | 0.331 |
| 30 | 0.244 | 0.277 |
| 40 | 0.227 | 0.245 |

使用データは `src/generate_item_banks_3pl.R` で生成した `data/3PL/`（500項目 × 10バンク）。MFI の結果は `data/3PL/results/` に保存。各実験の結果は `EXP***/results/` に保存。

## 元論文コードの問題点

`Train and test DQN on the simulated banks.py`（プロジェクトルートに配置）は元論文のオリジナルコードで、以下の問題がある：

- パスが `E:/...` の Windows パスにハードコードされており、そのままでは動かない
- `RESPOND` の乱数生成が `np.random.rand(1)`（スカラー1個）のため、全項目が同じ正誤になるバグがある
- `Choose_Action` の ε-greedy 判定に `np.random.randn()`（標準正規分布）を使っており、`np.random.rand()`（一様分布）が正しい
- `action_space = 500` がハードコードされており、バンクサイズと一致しない場合がある

EXP007 以降でこれらを修正済み。

## 各実験の説明

### EXP007 — DQN ベースライン（元論文再現・ローカル動作版）

`EXP007/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

元論文コードのパス問題・バグを修正しローカル/Colab で動くよう整備。アルゴリズム本体は元論文と同一。使用バンク：`data/uncorrelated_banks/`（200項目）。

### EXP008 — TD ターゲット計算の修正

`EXP008/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

EXP007 の replay buffer 学習で出題済み項目が Q(next) の候補に含まれる問題を修正。`terminal` フラグと `next_available` マスク（`masked_fill(-inf)`）を追加。使用バンク：`data/uncorrelated_banks/`（200項目）。

### EXP009 — data/3PL バンクへの切り替え

`EXP009/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 のアルゴリズムを維持しつつ `data/3PL/`（500項目）に変更。元論文準拠のバンク設定により MFI の RMSE が元論文値に近づいた。

### EXP010 — 全修正統合版（最新・これを参照）

`EXP010/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 に対して RESPOND バグ修正・TD ターゲットマスク修正（EXP008 由来）・data/3PL 使用（EXP009 由来）を統合。元論文再現実験として最も修正が揃ったバージョン。

| 変更内容 | EXP007 | EXP008 | EXP009 | EXP010 |
|---|---|---|---|---|
| RESPOND バグ修正 | ✓ | ✓ | ✓ | ✓ |
| TD ターゲットのマスク修正 | — | ✓ | — | ✓ |
| data/3PL 使用 | — | — | ✓ | ✓ |

使用バンク：`data/3PL/`（500項目）。結果保存先：`EXP010/results/`。

## 現状の再現性評価

### MFI の RMSE 比較

| Test length | 元論文 (MFI) | EXP007 (uncor, 200項目) | EXP009/010 (3PL, 500項目) |
|---:|---:|---:|---:|
| 10 | 0.493 | 0.701 | 0.504〜0.522 |
| 20 | 0.331 | 0.462 | 0.327〜0.340 |
| 30 | 0.277 | 0.377 | 0.271〜0.278 |
| 40 | 0.245 | 0.336 | 0.240〜0.245 |

EXP009・EXP010 は step 40 でほぼ一致するが、序盤（step 10〜20）では依然として乖離が残っている。

## リポジトリ構成（要点）

- `Train and test DQN on the simulated banks.py` — 元論文のオリジナル DQN コード（バグあり）
- `EXP007/` — DQN ベースライン（200項目バンク）
- `EXP008/` — TD ターゲット修正（200項目バンク）
- `EXP009/` — 3PL バンク切り替え（500項目）
- `EXP010/` — 全修正統合版（500項目、最新）
- `src/` — 共有コード（`models.py`, `utils.py`, `Functions.R`, `generate_item_banks_3pl.R`, `Test MFI on the 3PL bank.R` 等）
- `data/3PL/` — 元論文準拠バンク（500項目 × 10バンク、再現実験の使用データ）
- `papers/` — 参考論文 PDF・Markdown 版
- `refactoring.md` — 本ディレクトリの目的・各実験の説明・再現性評価の詳細記録

## 実行コマンド

```bash
# Python 依存関係
uv sync

# リント・型チェック
uv run ruff check -- src
uv run ruff format --check -- src
uv run pyright

# R スクリプト（ファイル名スペースに注意）
Rscript "src/generate_item_banks_3pl.R"
Rscript "src/Test MFI on the 3PL bank.R"
```
