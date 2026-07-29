# EXP018: DQN with Posterior Variance State（PyMC ADVI）

## 目的

EXP017 との比較実験。事後分散の計算方法を EAP 求積（EXP017）から **PyMC の平均場 ADVI** に切り替え、同じ状態表現（θ̂ + 事後分散）で挙動を比較する。

## EXP007 / EXP017 との変更点

| 変更内容 | EXP007 | EXP017 | EXP018 |
|---|---|---|---|
| 状態の次元 | 1（θ̂ のみ） | 2（θ̂ + 事後分散） | 2（θ̂ + 事後分散） |
| θ推定・分散の計算方法 | MLE + ヒューリスティック | EAP 求積（61点） | PyMC 平均場 ADVI |
| 全正解/全不正解の処理 | 端点ヒューリスティック | EAP で自然に解決 | ADVI で自然に解決 |

## ADVI の位置づけ

PyMC の `pm.fit(method="advi")` は **平均場変分推論（Mean-Field VI）** を実行する。

- 事後分布 $p(\theta|\mathbf{x})$ を正規分布 $q(\theta) = N(\mu_q, \sigma_q^2)$ で近似する
- ELBO（証拠下界）を勾配法で最大化し、$\mu_q$（事後平均）と $\sigma_q$（事後標準偏差）を得る
- MCMC より大幅に高速で、EAP 求積と同様に確定的（ランダム誤差なし）
- 1次元 3PL の場合、事後分布はほぼ単峰なので ADVI の近似精度は十分に高い

## 状態の定義

| タイミング | theta_hat | post_var |
|---|---|---|
| 初期 | Uniform(-0.5, 0.5) | 1.0（事前分散） |
| ステップ i 後 | ADVI posterior mean μ_q | ADVI posterior variance σ_q² |

## ADVI 設定

| パラメータ | 値 | 備考 |
|---|---|---|
| `n_advi_iter` | 2000 | 遅すぎる場合は 500〜1000 に下げる |
| 事前分布 | N(0, 1) | EXP017 と同一 |
| 最適化器 | ADAM（PyMC デフォルト） | |

## 使用バンク

`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）

EXP007・EXP017 と同一バンクで比較する。

## 結果保存先

- DQN 結果: `EXP018/results/`
- 比較対象: `EXP007/results/`（MFI）、`EXP017/results/`（EAP 求積版）

## 主要結果

（実験後に記入）

| Test length | MFI（EXP007） | EXP007 DQN γ=0.1 | EXP017 DQN γ=0.1 | EXP018 DQN γ=0.1 |
|---:|---:|---:|---:|---:|
| 10 | 0.701 | 0.653 | - | - |
| 20 | 0.462 | 0.443 | - | - |
| 30 | 0.377 | 0.367 | - | - |
| 40 | 0.336 | 0.329 | - | - |

## 速度の目安

ADVI は MCMC より速いが、EAP 求積（∼0.001秒/回）よりは遅い。

| フェーズ | 計算量 | 目安（n_advi_iter=2000） |
|---|---|---|
| TRAIN | 1000受検者 × 40ステップ = 40,000 回 | 数時間（Colab T4） |
| Validation | 200受検者 × 40ステップ × 20回 = 160,000 回 | TRAIN に含む |
| TEST | 5000受検者 × 40ステップ = 200,000 回 | 数時間 |

時間が掛かりすぎる場合は `n_advi_iter=500`・`training_size=500`・`validation_size=100` への削減を検討する。
