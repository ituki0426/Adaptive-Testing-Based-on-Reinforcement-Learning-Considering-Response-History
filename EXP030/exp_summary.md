# EXP030: 全経路で厳密な出題済み項目マスクを適用するDQN

## 目的

EXP008ではReplay buffer内の`Q(next)`に対して出題済み項目を除外したが、
学習時のgreedy選択とvalidation/testでは出題済み項目のQ値を`0`にしていた。
また、学習時のランダム探索は`any(item_id)`で分岐していたため、項目ID 0だけが
出題済みの場合に項目0を再選択しうる。

EXP030では、`deep_CAT`の行動マスクと同様に、学習・評価のすべての経路で
出題済み項目を候補集合から厳密に除外する。

## Notebook

`EXP030/notebook/Train_and_test_DQN_on_the_simulated_banks.ipynb`

## EXP008からの変更

行動制約以外のアルゴリズムと実験条件はEXP008を維持する。

- 単一受検者用の`Available_Action_Mask`を追加する。
- validation/test用の`Available_Action_Mask_Test`を追加し、受検者ごとに異なる
  出題履歴から2次元マスクを構築する。
- 学習時のgreedy選択では、出題済み項目のQ値を`0`ではなく`-inf`にする。
- 学習時のランダム探索では、`np.flatnonzero(available)`で得た未出題項目集合から
  直接サンプリングする。`any(item_id)`は使用しない。
- validation/testでも、受検者ごとに出題済み項目のQ値を`-inf`にする。
- Replay bufferへ保存する`next_available`も同じ`Available_Action_Mask`から生成する。
- 学習・評価の選択後に、選択項目がマスク上で利用可能だったことをassertする。
- モデルと結果の保存先を`EXP030/models/`、`EXP030/results/`へ分離する。
- Colab以外でも先頭セルが停止しないよう、Google DriveのmountをImportErrorで保護する。

行動制約の適用箇所は次のとおりである。

| 経路 | EXP030の処理 |
|---|---|
| 学習時greedy | 出題済みQ値を`-inf`にしてargmax |
| 学習時探索 | 未出題項目ID集合から直接サンプリング |
| Replay buffer | 遷移ごとの`next_available`を保存 |
| TDターゲット | `Q(next)`を`next_available`で`-inf`マスク |
| 終端遷移 | `terminal`で`Q(next)=0` |
| validation | 受検者ごとの出題済みQ値を`-inf`にしてargmax |
| test | 受検者ごとの出題済みQ値を`-inf`にしてargmax |

## 維持する実験条件

- DQN種別: 通常DQN。Double DQNには変更しない。
- データセット: `data/uncorrelated_banks/`（分散値に`sqrt`を取らずに生成）
- 既定のバンク: `item_bank_uncor_1.csv`
- 項目数: 500
- テスト長: 40
- 状態: 現在のMLE能力推定値`theta_hat`（1次元）
- 報酬: 学習受検者の真のthetaにおける選択項目のFisher情報量
- 学習theta: `N(0, 1)`（`prior="normal"`）
- discount factor: `gamma=0.1`
- epsilon-greedy: `epsilon=0.1`
- replay memory: 1,000
- batch size: 128
- target network更新間隔: 40 learning steps
- 正値パラメータ制約: EXP008の`Apply_Positive_Constraint`を維持
- 応答生成: 3PLから項目ごとに独立した乱数で生成
- 能力推定: `[-4, 4]`の有界MLE、尤度内の確率を
  `[1e-10, 1-1e-10]`にclip

学習開始時のNumPyおよびPyTorchのグローバルseedは、EXP008と同様に固定していない。
したがってEXP008との性能比較では複数seedによる反復実験が必要である。

## 保存先

- モデル: `EXP030/models/dqn_{prior}_{bank_type}_{bank_id}_gamma_{gamma}.pt`
- 受検者別結果: `EXP030/results/records_{bank_type}_{bank_id}_DQN_{prior}_gamma_{gamma}.csv`
- ステップ別集計: `EXP030/results/summary_{bank_type}_{bank_id}_DQN_{prior}_gamma_{gamma}.csv`

同じ`data/uncorrelated_banks/`上のMFI比較結果は`EXP007/results/`を使用する。

## 検証状況

- Notebook JSONの妥当性を確認した。
- 全コードセルのPython構文を確認した。
- 負のQ値を返すネットワークでも出題済み項目を再選択しないことを確認した。
- 項目ID 0だけが出題済みの状態でランダム探索を500回実行し、項目0を
  再選択しないことを確認した。
- 複数受検者が異なる出題履歴を持つvalidation/test用マスクを確認した。
- 20項目、テスト長4、学習受検者4人の縮小設定で、Replay buffer更新、
  TDターゲット計算、validationまでのスモークテストを完了した。

既定値による全規模学習とRMSE評価は未実行である。結果を解釈する際は、DQNの
RMSEだけで判断せず、同じデータセット上のMFIとstep 10、20、30、40で比較する。
