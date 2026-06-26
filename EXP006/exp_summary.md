# EXP006: DQN with MAP state and Laplace posterior variance

## 目的

`EXP004` の状態 `[θ̂, var]` を、`MLE + 1 / (I(θ̂) + 1)` から
`MAP + (負の対数事後の2階微分)^{-1}` に置き換える。

## 実装方針

- 状態は 2 次元 `[theta_map, var_map]`
- 能力推定は `N(0, 1)` 事前の MAP 推定
- 分散は MAP 点での負の対数事後の数値2階微分の逆数
- 全問正答・全問誤答でもヒューリスティックは使わず、MAP を直接解く
- データ生成の `prior` 引数は従来通り受検者母集団の生成用で、推定時の事前分布とは分離

## 主な差分

- `NEG_LOG_POSTERIOR(...)` を追加
- `MLE(...)` / `MLE_TEST(...)` を `MAP(...)` / `MAP_TEST(...)` に置換
- `POSTERIOR_VAR(...)` は `NEG_LOG_POSTERIOR` の中心差分2階微分から算出
- 保存名は `dqn_maphess_*` / `DQNmaphess_*` に変更

## 対象ファイル

- `EXP006/src/Train and test DQN on the simulated banks.py`
