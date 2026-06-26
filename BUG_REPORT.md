# BUG_REPORT: EXP003 DQN RMSE が不自然にガタつく問題

## 概要

`EXP003/result/summary_uncor_1_DQN_normal_gamma_0.1.csv` と
`EXP003/result/summary_uncor_1_MFI.csv` を比較すると、DQN の RMSE が MFI より
不自然に大きく上下している。

確認した限り、summary CSV の集計ミスではなく、DQN 側のシミュレーション実装に
起因する可能性が高い。

## 対象ファイル

- `EXP003/src/Train and test DQN on the simulated banks.py`
- `EXP003/src/Test MFI on the simulated bank.R`
- `EXP003/result/summary_uncor_1_DQN_normal_gamma_0.1.csv`
- `EXP003/result/summary_uncor_1_MFI.csv`
- `EXP003/result/records_uncor_1_DQN_normal_gamma_0.1.csv`
- `EXP003/result/records_uncor_1_MFI.csv`

## 確認した症状

records から summary を再集計したところ、既存 summary と一致した。
したがって、CSV の集計処理自体が原因ではない。

DQN の序盤では、step ごとの回答率が極端に振れていた。

```text
DQN step1  resp_rate = 0.347
DQN step2  resp_rate = 0.987
DQN step3  resp_rate = 0.019
DQN step5  resp_rate = 1.000
DQN step10 resp_rate = 1.000
```

一方、MFI の回答率は序盤でもおおむね 0.57 から 0.67 程度で推移していた。

また、DQN は各 step の選択アイテムが少数に集中していた。

```text
DQN step1: unique items = 1
DQN step2: unique items = 3
DQN step3: unique items = 4
```

特に DQN step1 では、5000人全員が item 94 を選択していた。

## 主原因: RESPOND の乱数生成がベクトル化に対応していない

`EXP003/src/Train and test DQN on the simulated banks.py` の `RESPOND()` は以下の実装になっている。

```python
def RESPOND(item_para, theta, D=1):
    a = item_para[:,0]
    b = item_para[:,1]
    c = item_para[:,2]
    p = (1-c) / (1 + np.exp(-D*a*(theta-b))) + c
    resp = (np.random.rand(1) <= p).astype(int)
    return resp
```

学習時に 1人分だけ処理する場合は大きな問題になりにくいが、validation/test では
`p` が複数受検者分のベクトルになる。このとき `np.random.rand(1)` により乱数が1個だけ
生成され、その1個の乱数が全受検者の確率ベクトルと比較される。

その結果、同一 step 内の全受検者の回答が同じ乱数に強く依存し、回答が過度に相関する。
これにより、step ごとの正答率と推定値が不自然に振れ、RMSE 曲線がガタつく。

MFI の R 実装では、`genPattern()` を受検者ごとに呼んでいるため、DQN と MFI で回答生成の
乱数構造が一致していない。

## 修正案

`RESPOND()` で確率 `p` と同じ shape の乱数を生成する。

```python
resp = (np.random.random(size=p.shape) <= p).astype(int)
```

修正後の例:

```python
def RESPOND(item_para, theta, D=1):
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    p = (1 - c) / (1 + np.exp(-D * a * (theta - b))) + c
    resp = (np.random.random(size=p.shape) <= p).astype(int)
    return resp
```

## 追加で確認した問題: epsilon-greedy の探索確率

同じ DQN スクリプトの `Choose_Action()` では、epsilon-greedy が以下のように実装されている。

```python
if np.random.randn() >= epsilon:
```

`epsilon = 0.1` の場合、本来は約10%をランダム行動にする意図だと考えられる。
しかし標準正規乱数 `np.random.randn()` を使っているため、ランダム行動の確率は約54%になる。

通常の epsilon-greedy として扱うなら、以下に修正する。

```python
if np.random.rand() >= epsilon:
```

この問題は RMSE 曲線のガタつきの直接原因というより、DQN の学習条件を意図と異なるものにする
実装上の問題である。

## 影響範囲

少なくとも以下の DQN 結果は、現在のまま MFI と公平比較しない方がよい。

- `EXP003/result/records_uncor_1_DQN_normal_gamma_0.1.csv`
- `EXP003/result/summary_uncor_1_DQN_normal_gamma_0.1.csv`

同じ `RESPOND()` 実装を使っている他の DQN 系スクリプトにも同種の問題がある可能性がある。
検索時点では以下でも同じ `np.random.rand(1)` 実装が見つかった。

- `src/utils.py`
- `EXP004/src/Train and test DQN on the simulated banks.py`
- `EXP005/src/Train and test DQN on the simulated banks.py`
- `EXP006/src/Train and test DQN on the simulated banks.py`

## 必要な対応

1. `RESPOND()` を `p.shape` に合わせた乱数生成に修正する。
2. DQN の epsilon-greedy を通常の定義に合わせるか、現在の約54%探索を意図した条件として明記する。
3. 修正後に DQN を training/validation/model selection から再実行する。
4. `records_*_DQN_*.csv` と `summary_*_DQN_*.csv` を再生成する。
5. 再生成後の結果を MFI/FIWL/DRQN と比較し直す。

## 注意

`RESPOND()` を修正すると validation/test だけでなく training 中の挙動も変わる可能性がある。
そのため、既存モデルをそのまま使って test だけ再実行するより、学習からやり直す方が比較として安全である。
