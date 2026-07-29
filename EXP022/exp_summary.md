# EXP022: DQN with Posterior Variance State（PyMC ADVI、Mac ローカル最適化版）

## 目的

EXP018 の PyMC ADVI アプローチをそのまま維持しつつ、Mac ローカル環境で実用的に動作するよう最適化した実験。Colab 依存コードをすべて除去し、並列化と ADVI パラメータ削減により実行時間を短縮する。

## EXP018 からの変更点

| 変更内容 | EXP018（Colab 想定）| EXP022（Mac ローカル）|
|---|---|---|
| Google Drive マウント | あり | **なし** |
| PyMC pip インストール | あり | **なし** |
| Colab パス | `find_project_root()` に含む | **削除**（`cwd` の親をたどるのみ） |
| PYTENSOR_FLAGS | 未設定 | **`FAST_RUN,floatX=float64,allow_gc=False`** |
| デバイス | `cuda` or `cpu` | **`mps` or `cpu`**（Apple Silicon 対応） |
| `n_advi_iter` | 2000 | **500**（1次元問題には十分） |
| `n_advi_samples` | 500 | **200** |
| validation/test のループ | 逐次（受検者ごとに for）| **`ProcessPoolExecutor(fork)` で並列化** |
| `n_jobs` | なし | **-1（全コア）/ 1（逐次・デバッグ用）** |

## 並列化の設計

訓練ループは受検者を 1 人ずつ逐次処理（前ステップの状態に依存するため並列化不可）。

validation と test は、各ステップで受検者間が独立なため ADVI を並列実行する：

```
parallel_advi(item_bank, item_id_mat, resp_mat, n_subjects, ...)
→ ProcessPoolExecutor(fork, max_workers=os.cpu_count())
→ [_advi_worker(item_paras_s, resp_s, ...) for s in range(n_subjects)]
```

`fork` コンテキストを使うことで PyMC の pytensor グラフを子プロセスに引き継げる。PyMC がスレッドを立ち上げる前に fork するため Mac ローカルでは安全。

問題が起きる場合は `n_jobs=1` で逐次動作に切り替えられる。

## 実験条件（EXP018 から変更なし）

| パラメータ | 値 |
|---|---|
| 状態 | `[θ̂_ADVI, post_var_ADVI]`（次元=2） |
| ネットワーク | 2 → 50 → 30 → 500 |
| gamma | 0.1 |
| training_size | 1000 |
| validation_size | 200 |
| 事前分布 | N(0, 1) |

## 使用バンク

`data/uncorrelated_banks/`（500項目、分散値に `sqrt` を取らずに生成）

EXP007・EXP017・EXP018 と同一バンク。

## 結果保存先

- DQN 結果: `EXP022/results/`
- 比較対象:
  - MFI: `EXP007/results/`
  - EXP007 DQN（MLE）: `EXP007/results/`
  - EXP017 DQN（EAP + var）: `EXP017/results/`

## 主要結果

（実験後に記入）

| Test length | MFI | EXP007 DQN（MLE）| EXP017 DQN（EAP+var）| EXP022 DQN（ADVI）|
|---:|---:|---:|---:|---:|
| 10 | 0.701 | 0.653 | 0.502 | - |
| 20 | 0.462 | 0.443 | 0.380 | - |
| 30 | 0.377 | 0.367 | 0.326 | - |
| 40 | 0.336 | 0.329 | 0.294 | - |
