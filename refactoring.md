本ディレクトリの目的

元論文：papers/s13428-024-02498-x.pdf

元論文におけるDQN(normal)とMFIのシミュレーション実験を再現すること。

元論文のテーブル1

| Test length | DQN (normal) | MFI |
|---:|---:|---:|
| 10 | 0.415 | 0.493 |
| 20 | 0.278 | 0.331 |
| 30 | 0.244 | 0.277 |
| 40 | 0.227 | 0.245 |

本リポジトリでは、非相関アイテムバンクとして 2 種類のデータセットを使っている。

| データセット | 項目数 | 分散指定の扱い | 主な使用実験 | 位置づけ |
|---|---:|---|---|---|
| `data/uncorrelated_banks/` | 500 | 分散値に `sqrt` を取らずに生成 | EXP007, EXP008, EXP011 | 初期のローカル再現実験で使用したバンク |
| `data/3PL/` | 500 | 分散値に `sqrt` を取り、標準偏差として生成 | EXP009, EXP010 | 元論文 Table 1 の再現により近いバンク |

`data/3PL/` は `src/generate_item_banks_3pl.R` によって生成された `data/3PL/item_bank_uncor_1.csv`〜`data/3PL/item_bank_uncor_10.csv` を使用する。元論文のパラメータ設定では a, b, c のばらつきは分散として記述されているため、乱数生成時には `sqrt(分散)` を標準偏差として渡す必要がある。`data/uncorrelated_banks/` はこの `sqrt` 処理を行っていないため、項目パラメータの分布が `data/3PL/` と異なる。

MFI の結果は、評価対象のデータセットごとに保存先が異なる。

| MFI の対象データセット | 分散指定の扱い | MFI 結果保存先 |
|---|---|---|
| `data/uncorrelated_banks/` | 分散値に `sqrt` を取らずに生成 | `EXP007/results/` |
| `data/3PL/` | 分散値に `sqrt` を取り、標準偏差として生成 | `data/3PL/results/` |

DQN の各実験結果は `EXP(実験番号)/results/` 内に保存。

元論文のDQNのプログラムはTrain and test DQN on the simulated banks.py


## 各実験の説明

### EXP007

**DQN ベースライン（元論文再現・ローカル動作版）**

`EXP007/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

元論文コードをローカル環境・Colab で動くよう整備した実験。アルゴリズム本体は元論文と同一。

元論文コードからの主な変更点：
- `E:/...` のハードコードパスを `pathlib.Path` + `find_project_root()` でポータブルに解決
- ハイパーパラメータを `@dataclass Config` に一元化
- `RESPOND` の乱数生成バグを修正（`np.random.rand(1)` → `np.random.random(size=p.shape)`）
- `Choose_Action` の乱数を `np.random.randn()` → `np.random.rand()` に修正
- `action_space` をバンクサイズから自動取得
- ベストモデルを `copy.deepcopy(state_dict)` でメモリ保持し、訓練終了後に保存
- モデル選択基準を RMSE のみに統一
- ステップ別 summary CSV を追加出力

使用バンク：`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）
DQN 結果保存先：`EXP007/results/`
詳細：`EXP007/exp_summary.md`

### EXP008

**TD ターゲット計算の修正**

`EXP008/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

EXP007 の replay buffer 学習で Q(next) を計算する際に出題済み項目が候補に含まれてしまう問題を修正。

主な変更点：
- `RESPOND` の乱数生成を項目ごとに独立した乱数へ修正
- `MLE` / `MLE_TEST` に `np.clip(p, 1e-10, 1 - 1e-10)` を追加し、`log(0)` を防止
- memory に `terminal` フラグと `next_available` マスク（size=action_space）を追加
- Q(next) を `masked_fill(-inf)` で出題済み項目を除外してから argmax
- 終端ステップでは Q_next をゼロにして TD ターゲットを計算
- ベストモデルは `state_dict` をメモリ保持し、訓練終了後に `.pt` として保存

使用バンク：`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）
DQN 結果保存先：`EXP008/results/`
比較用プロット：`EXP008/notebook/plot_rmse_comparison.ipynb`
詳細：`EXP008/exp_summary.md`

主要結果（bank 1、RMSE）：

| γ | step 10 | step 20 | step 30 | step 40 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.636 | 0.449 | 0.376 | 0.341 |
| 0.3 | 0.662 | 0.465 | 0.385 | 0.351 |

EXP007 と比較すると、TD ターゲットのマスク修正だけでは DQN の明確な改善は確認できなかった。γ=0.1 の step 40 RMSE は EXP007 の 0.329 に対して EXP008 は 0.341 で悪化しており、DQN が MFI を上回る元論文結果の再現にはつながっていない。

### EXP009

**data/3PL バンクへの切り替え**

`EXP009/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 のアルゴリズムをそのまま維持しつつ、バンクを `data/3PL/`（500項目、分散値に `sqrt` を取って生成）に変更した実験。

主な変更点：
- バンクディレクトリを `data/3PL/` に固定
- ファイル名を `item_bank_uncor_{id}.csv` に合わせて変更
- `Config.bank_type` を削除（uncor 固定）
- n_items=500

使用バンク：`data/3PL/`（`src/generate_item_banks_3pl.R` で生成、500項目、分散値に `sqrt` を取って生成）
DQN 結果保存先：`EXP009/results/`
詳細：`EXP009/exp_summary.md`

### EXP010

**RESPOND 修正 + TD ターゲット修正 + data/3PL**

`EXP010/notebook/Train_and_test_DQN_on_the_3PL_banks.ipynb`

EXP007 に対して RESPOND バグ修正・TD ターゲット修正（EXP008 由来）・data/3PL への切り替え（EXP009 由来）を同時に適用した実験。元論文の再現実験として最も修正が揃ったバージョン。

| 変更内容 | EXP007 | EXP008 | EXP009 | EXP010 |
|---|---|---|---|---|
| RESPOND バグ修正 | ✓ | ✓ | ✓ | ✓ |
| TD ターゲットのマスク修正 | — | ✓ | — | ✓ |
| data/3PL（sqrt あり）使用 | — | — | ✓ | ✓ |

使用バンク：`data/3PL/`（500項目、分散値に `sqrt` を取って生成）
DQN 結果保存先：`EXP010/results/`
比較用プロット：`EXP010/notebook/plot_rmse_comparison.ipynb`（MFI vs DQN を gamma ごとに別プロット）
詳細：`EXP010/exp_summary.md`

### EXP011

**DQN with Estimated θ̂ Reward（論文アルゴリズム準拠）**

`EXP011/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

EXP007 との差は、報酬の計算に使う θ の違いのみ。EXP007〜EXP010 では元論文コードに合わせて真の θ で Fisher 情報量を計算していたが、元論文 Algorithm 1 と Eq. 13 では推定値 θ̂_l で計算した `I_i_l(θ̂_l)` が報酬として定義されている。

主な変更点：
- TRAIN 関数内の報酬計算を `FI(item_bank[action,], training_theta[j])` から `FI(item_bank[action,], state[-1:])` に変更
- ネットワーク構造、バンク、prior、受検者数、テスト長などは EXP007 と同一
- γ=0.1, 0.3, 0.9 を実施

使用バンク：`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）
DQN 結果保存先：`EXP011/results/`
比較用プロット：`EXP011/notebook/plot_rmse_comparison.ipynb`
詳細：`EXP011/exp_summary.md`

主要結果（bank 1、RMSE）：

| γ | step 10 | step 20 | step 30 | step 40 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.626 | 0.442 | 0.370 | 0.330 |
| 0.3 | 0.700 | 0.481 | 0.398 | 0.359 |
| 0.9 | 0.934 | 0.694 | 0.553 | 0.482 |

EXP007 との比較では、γ=0.1 の step 40 RMSE は EXP007 の 0.329 に対して EXP011 は 0.330 でほぼ同等。γ=0.3 は EXP007 の 0.342 に対して EXP011 は 0.359 と悪化。γ=0.9 は EXP007 の 0.487 に対して EXP011 は 0.482 とわずかに良化したが、γ=0.1 より大幅に悪い。

## 現状の再現性評価

### 結論

現時点では、元論文 Table 1 の **DQN(normal) が MFI より低い RMSE を示す** という主要結果は再現できていない。

重要なのは、データセットによって再現できている部分が異なる点である。

- `data/uncorrelated_banks/`（sqrt なし）では、DQN が MFI よりやや低い RMSE になる場合がある。しかし MFI 自体が元論文より大幅に悪く、この条件は元論文 Table 1 のデータ設定を再現できていない。
- `data/3PL/`（sqrt あり）では、MFI は元論文にかなり近づく。しかしこの条件では、DQN は step 20〜40 で MFI より RMSE が高く、元論文の DQN 優位性を再現できていない。

したがって、現状は「MFI が元論文に近い条件では DQN が MFI に負ける」状態であり、元論文の結論は未再現である。

### MFI の絶対値再現

元論文 Table 1 の MFI RMSE と、各データセットでの MFI 実測値：

EXP007 列は `data/uncorrelated_banks/`（sqrt なし）に対する MFI で、結果は `EXP007/results/` に保存している。EXP009/010 列は `data/3PL/`（sqrt あり）に対する MFI で、結果は `data/3PL/results/` に保存している。

| Test length | 元論文 (MFI) | EXP007 (uncor, sqrt なし) | EXP009/010 (3PL, sqrt あり) |
|---:|---:|---:|---:|
| 10 | 0.493 | 0.701 | 0.504〜0.522 |
| 20 | 0.331 | 0.462 | 0.327〜0.340 |
| 30 | 0.277 | 0.377 | 0.271〜0.278 |
| 40 | 0.245 | 0.336 | 0.240〜0.245 |

この表から、`data/3PL/`（sqrt あり）は MFI の絶対値再現にはかなり近い。一方、`data/uncorrelated_banks/`（sqrt なし）は MFI が元論文より大幅に悪く、DQN と MFI の優劣を見ても元論文再現の根拠としては弱い。

### DQN と MFI の比較

元論文の主張は、同じテスト長で DQN(normal) の RMSE が MFI より低いことである。再現実験でも、DQN と MFI は同じデータセット上で比較する必要がある。

**sqrt なしデータセット（`data/uncorrelated_banks/`）**

| Test length | MFI | EXP007 DQN γ=0.1 | DQN の改善量（MFI - DQN） | 元論文の改善量 | 優劣（RMSE） |
|---:|---:|---:|---:|---:|---|
| 10 | 0.701 | 0.653 | 0.048 | 0.078 | DQN < MFI |
| 20 | 0.462 | 0.443 | 0.019 | 0.053 | DQN < MFI |
| 30 | 0.377 | 0.367 | 0.010 | 0.033 | DQN < MFI |
| 40 | 0.336 | 0.329 | 0.007 | 0.018 | DQN < MFI |

sqrt なしでは DQN が MFI より低い RMSE を示すが、改善量は元論文 Table 1 よりかなり小さい。例えば step 20 では元論文の改善量が 0.053 であるのに対し、EXP007 では 0.019 にとどまる。さらに MFI の RMSE 自体も元論文から大きく外れているため、この結果だけでは元論文 Table 1 の再現とは言えない。

**sqrt ありデータセット（`data/3PL/`）**

| Test length | 元論文 DQN | 元論文 MFI | local MFI | EXP010 DQN best | 優劣（RMSE） |
|---:|---:|---:|---:|---:|---|
| 10 | 0.415 | 0.493 | 0.504 | 0.470 | DQN < MFI |
| 20 | 0.278 | 0.331 | 0.340 | 0.359 | DQN > MFI |
| 30 | 0.244 | 0.277 | 0.278 | 0.303 | DQN > MFI |
| 40 | 0.227 | 0.245 | 0.245 | 0.277 | DQN > MFI |

`EXP010 DQN best` は、bank 1 で実施した γ=0.05, 0.1, 0.3, 0.5, 0.7 のうち、各 step で最も低い RMSE を示した値である。最も有利な値を選んでも、step 20〜40 では DQN が MFI より悪い。特に step 40 では、元論文は DQN 0.227 < MFI 0.245 だが、再現実験では DQN 0.277 > MFI 0.245 となっている。

### 現状の問題点

**EXP007（data/uncorrelated_banks、500項目、sqrt なし）**
- DQN は MFI よりやや低い RMSE を示すが、MFI 自体が元論文より大幅に悪い
- アイテムバンク生成時に分散値へ `sqrt` を取っておらず、元論文の想定より項目パラメータのばらつきが大きいことが原因と考えられる
- この条件で DQN が MFI を上回っても、元論文 Table 1 の再現とは言いにくい

**EXP008（data/uncorrelated_banks、500項目、sqrt なし）**
- TD ターゲット計算で出題済み項目を除外する修正を入れたが、DQN の明確な改善は確認できなかった
- γ=0.1 の step 40 RMSE は EXP007 の 0.329 に対して EXP008 は 0.341 で、むしろ悪化した
- この結果から、replay buffer 学習時の Q(next) 候補に出題済み項目が含まれる問題だけでは、元論文 Table 1 との差は説明できない

**EXP009・EXP010（data/3PL、500項目、sqrt あり）**
- `src/generate_item_banks_3pl.R` で元論文の分散設定（a の分散 0.25、b の分散 1、c の分散 0.02）に対して `sqrt` を取り、標準偏差として乱数生成に渡したことで、MFI の RMSE が元論文の値に近づいた
- step 40 では元論文の 0.245 に対して 0.240〜0.245 とほぼ一致
- しかし DQN は step 20〜40 で MFI より RMSE が高く、元論文の DQN 優位性を再現できていない
- 現状の最重要課題は、MFI が元論文に近い `data/3PL/` 条件で DQN がなぜ MFI に負けるのかを調べること

**EXP011（data/uncorrelated_banks、500項目、sqrt なし）**
- 論文の定式化どおり推定 θ̂ で報酬を計算しても、EXP007 に対する明確な改善は確認できなかった
- γ=0.1 は EXP007 とほぼ同等、γ=0.3 は悪化、γ=0.9 はわずかに良化したが絶対性能は低い
- この結果から、元論文コードと論文アルゴリズムの報酬定義の乖離だけでは、元論文 Table 1 との差は説明できない

## 今後の確認事項

DQN が MFI を安定して上回るという元論文 Table 1 の結果は、現時点では再現できていない。そのため、追加実験の前に以下を確認する必要がある。

- 元論文本文のアルゴリズム・数式・実験条件と、公開されている実装プログラムの間に乖離がないか確認する
- 公開実装プログラムに、再現結果へ影響するバグが残っていないか確認する
- 乖離やバグを見つけた場合は、それが「論文本文準拠の修正」なのか「公開コード準拠の再現」なのかを区別して記録する
- 修正後は、DQN の RMSE だけでなく MFI との優劣を step 10, 20, 30, 40 で比較する
