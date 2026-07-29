# EXP028: DQN on real responses (EXP027-compatible)

## 目的

`EXP027/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb` を基礎として、
シミュレーション応答を実回答データへ置き換えた DQN 実験を実装する。

参照した元論文添付コードは、リポジトリ直下の
`Train and test DQN on the real responses.py` である。
`EXP027/notebook/Train_and_test_DQN_on_the_real_responses.ipynb` は参照していない。

## Notebook

`EXP028/notebook/Train_and_test_DQN_on_the_real_responses.ipynb`

## データ

- データセット: `data/LNIRT_CredentialForm1/`
- 項目数: 170
- 元の訓練ファイル: 1,308 人
- DQN 訓練: 1,000 人
- 検証: 200 人
- テスト: 328 人
- 訓練と検証は `seed=20260430` による固定 permutation から非重複に分割する
- 元の訓練ファイルのうち、訓練・検証に割り当てられなかった 108 人は使用しない

## EXP027 から維持した条件

- 状態: 現在の能力推定値 `theta_hat` 1次元
- 能力推定: MLE / EAP を `Config.estimation_method` で切替可能
- 既定の能力推定: MLE
- EAP: 一様事前分布 `U(-4, 4)`、62求積点
- テスト長: 40
- discount factor: `gamma=0.1`
- epsilon-greedy: `epsilon=0.1`
- replay memory: 1,000
- batch size: 128
- target network 更新間隔: 40 learning steps
- TD ターゲットでは出題済み項目を action mask で除外する
- 終端 transition の `Q(next)` は 0 とする
- 検証時の step 7 から 40 の平均 RMSE が最小のモデルを保存する

## 実データ化による変更

- 訓練、検証、テストの応答は CSV に記録された実回答から取得する
- 訓練報酬は EXP027 準拠で、各訓練受検者の true theta における選択項目の
  Fisher 情報量とする
- 元論文添付の実データコードにある、現在の推定 theta で報酬を計算する定義は
  採用しない
- 検証データは訓練データと非重複にし、検証リークを避ける
- action space は item bank の行数から自動取得する
- モデルは `EXP028/models/`、評価結果は `EXP028/results/` に保存する

## 注意点

`LNIRT_CredentialForm1` の difficulty parameter `b` には約 -65.9 から 65.6 の
値が含まれる。EXP027 と同じ MLE の全正答・全誤答時処理は、現在値と item bank
全体の `b` の最大値または最小値との中点を使う。このため、テスト初期の能力推定値と
RMSE が非常に大きくなる場合がある。本実験では EXP027 との条件対応を優先し、
この処理を変更していない。

## 検証状況

既定値の全規模実験は未実行である。縮小設定
（訓練4人、検証4人、テスト長3、CPU）では、データ読込、訓練、検証、テスト、
モデル保存、records CSV 保存、summary CSV 保存までのスモークテストを完了した。

実験結果を解釈する際は DQN の RMSE だけで判断せず、同じ
`data/LNIRT_CredentialForm1/` 上の MFI と比較する必要がある。
