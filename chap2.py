# %% import

from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib_fontja


# %% load

X, y = load_extended_boston()

# デフォルトでは75%が訓練データ
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% 単回帰

# numpyだとindiceにtupleが渡せる。以下だと全ての行の5列目だけ
# reshapeは行列のサイズをreshapeする。-1はもう一方に合わせてよしなに、ということなので、以下だと1列でよしなな行数になる
X_train_single = X_train[:,5].reshape(-1, 1)
X_test_single = X_test[:,5].reshape(-1, 1)

display(X_train_single)

# 学習
lm_single = linear_model.LinearRegression()
lm_single.fit(X_train_single, y_train)
y_pred_train = lm_single.predict(X_train_single)

# 散布図と回帰直線を描画
plt.xlabel('住居の平均部屋数')
plt.ylabel('住宅価格の中央値')
plt.scatter(X_train_single, y_train)
plt.plot(X_train_single, y_pred_train, color='red')

# 単回帰の結果
print(f'切片: {lm_single.intercept_:.2f}')
print(f'傾き: {lm_single.coef_[0]:.2f}')
print(f'訓練データの当てはまり: {lm_single.score(X_train_single, y_train):.2f}')
print(f'検証データの当てはまり: {lm_single.score(X_test_single, y_test):.2f}')
# print(f'{}')
# print(f'{}')
# print(f'{}')
# print(f'{}')


# %% 重回帰

lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
y_pred_train = lm.predict(X_train)

# 単回帰の結果
print(f'切片: {lm.intercept_:.2f}')
print(f'傾き: {lm.coef_}')
print(f'訓練データの当てはまり: {lm.score(X_train, y_train):.2f}')
print(f'検証データの当てはまり: {lm.score(X_test, y_test):.2f}')

