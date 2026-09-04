



テスト長を縮小40→10
シミュレーションデータ実験に絞る
項目バンク数もいくつか試してみる:[100, 200, 300, 400, 500]
それに伴い、モデルの更新基準も変更する。
今まで：step7~40までのRMSEの平均がbestより小さい場合に更新
変更後：step1~10までのRMSEの平均がbestより小さい場合に更新

$$
I_{\text{test}}(\hat\theta)=\sum_{i\in\mathcal A} I_i(\hat\theta)
$$

$$
\widehat{\mathrm{Var}}(\hat\theta)
\approx I_{\text{test}}(\hat\theta)^{-1},
\qquad
SE(\hat\theta)
\approx \frac{1}{\sqrt{I_{\text{test}}(\hat\theta)}}
$$

$$
state = [theta_hat_MLE,log(SE_MLE + ε),step / test_length]
$$

もしくは

$$
state = [theta_hat,log(SE + ε),step / test_length,mle_available]
$$