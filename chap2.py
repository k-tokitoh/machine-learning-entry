# %% import

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib_fontja

from sklearn import linear_model, preprocessing, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston
from mlxtend.plotting import plot_decision_regions


# %% load

X, y = load_extended_boston()

# デフォルトでは75%が訓練データ
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% 単回帰

# numpyだとindiceにtupleが渡せる。以下だと全ての行の5列目だけ
# reshapeは行列のサイズをreshapeする。-1はもう一方に合わせてよしなに、ということなので、以下だと1列でよしなな行数になる
X_train_single = X_train[:, 5].reshape(-1, 1)
X_test_single = X_test[:, 5].reshape(-1, 1)

display(X_train_single)

# 学習
lm_single = linear_model.LinearRegression()
lm_single.fit(X_train_single, y_train)
y_pred_train = lm_single.predict(X_train_single)

# 散布図と回帰直線を描画
plt.xlabel("住居の平均部屋数")
plt.ylabel("住宅価格の中央値")
plt.scatter(X_train_single, y_train)
plt.plot(X_train_single, y_pred_train, color="red")

# 単回帰の結果
print(f"切片: {lm_single.intercept_:.2f}")
print(f"傾き: {lm_single.coef_[0]:.2f}")
print(f"訓練データの当てはまり: {lm_single.score(X_train_single, y_train):.2f}")
print(f"検証データの当てはまり: {lm_single.score(X_test_single, y_test):.2f}")


# %% 重回帰

lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
y_pred_train = lm.predict(X_train)

# 単回帰の結果
print(f"切片: {lm.intercept_:.2f}")
print(f"傾き: {lm.coef_}")
print(f"訓練データの当てはまり: {lm.score(X_train, y_train):.2f}")
print(f"検証データの当てはまり: {lm.score(X_test, y_test):.2f}")

# これだと訓練データへの当てはまりが95%, 検証データの当てはまりが61%なので過学習している


# %% L2正則化 = Ridge回帰

ridge = linear_model.Ridge(alpha=1.0, random_state=0)
ridge.fit(X_train, y_train)
y_pred_train = ridge.predict(X_train)

# 単回帰の結果
print(f"切片: {ridge.intercept_:.2f}")
print(f"傾き: {ridge.coef_}")
print(f"訓練データの当てはまり: {ridge.score(X_train, y_train):.2f}")
print(f"検証データの当てはまり: {ridge.score(X_test, y_test):.2f}")


# %% L1正則化 = Lasso

lasso = linear_model.Lasso(alpha=0.01, max_iter=2000, random_state=0)
lasso.fit(X_train, y_train)
y_pred_train = lasso.predict(X_train)

# 単回帰の結果
print(f"切片: {lasso.intercept_:.2f}")
print(f"傾き: {lasso.coef_}")
print(f"パラメータが0でない特徴量の数: {np.count_nonzero(lasso.coef_)}")
print(f"訓練データの当てはまり: {lasso.score(X_train, y_train):.2f}")
print(f"検証データの当てはまり: {lasso.score(X_test, y_test):.2f}")

# %% ロジスティック回帰 > 模擬データの生成

# 乱数のシードを固定
np.random.seed(0)

# 2次元ガウス分布で模擬データ100人分を作成
mean = [10, 10]  # 平均値
cov = [[10, 3], [3, 10]]  # 分散共分散行列
x1, y1 = np.random.multivariate_normal(mean, cov, 100).T  # 2次元データ生成
true_false1 = np.random.rand(100) > 0.9  # 0-1の一様乱数の10%がTrue
label1 = np.where(true_false1, 1, 0)  # AdvancedindexingでLabelデータ生成

# 2次元ガウス分布で模擬データ100人分を作成
mean = [20, 20]  # 平均値
cov = [[8, 4], [4, 8]]  # 分散共分散行列
x2, y2 = np.random.multivariate_normal(mean, cov, 100).T
true_false2 = np.random.rand(100) > 0.1  # 01の一様乱数の90%がTrue
label2 = np.where(true_false2, 1, 0)  # AdvancedindexingでLabelデータ生成

# データを描画
X = np.r_[x1, x2]  # 配列の結合
Y = np.r_[y1, y2]
label = np.r_[label1, label2]

# ラベル1の継続会員とラベル0の退会会員をAdvancedindexingで取り出して描画
plt.scatter(
    X[label == 1],
    Y[label == 1],
    marker="^",
    s=30,
    color="blue",
    label="1:continue",
)
plt.scatter(
    X[label == 0],
    Y[label == 0],
    marker=",",
    s=30,
    color="red",
    label="0:withdraw",
)

plt.xlabel("年間購入回数")
plt.ylabel("平均購入単価")
plt.legend()
plt.show()


# %% ロジスティック回帰

# 訓練データとテストデータに分割
Data = np.c_[X, Y]
X_train, X_test, y_train, y_test = train_test_split(Data, label, random_state=0)

# ロジスティック回帰の適用
# clfはclassfierと思われる
lr = linear_model.LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

# 学習した識別平面とテストデータをプロットする
plot_decision_regions(X_test, y_test, clf=lr, legend=2)

print(f"正解率: {lr.score(X_test, y_test):.2f}")

print(lr.predict([[5, 20], [15, 20], [25, 20]]))
print(lr.predict([[5, 15], [15, 15], [25, 15]]))
print(lr.predict([[5, 10], [15, 10], [25, 10]]))
print(lr.predict([[5, 5], [15, 5], [25, 5]]))


# %%  SVC > 疑似データの生成

# データの作成
X = np.zeros((20, 2))
print(X)
X[0:10, 1] = range(0, 10)
X[10:20, 1] = range(0, 10)
X[0, 0] = 1.0
X[9, 0] = 1.0
X[1:9, 0] = 3.0
X[10:20, 0] = range(-1, -11, -1)
X[9, 0] = 1
X[19, 0] = -1
# X[19, 0] = 2
y = np.zeros((20))
y[10:20] = 1.0
y = y.astype(np.int8)

# 描画
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

sc = preprocessing.StandardScaler()
sc.fit(X)
X_std = sc.transform(X)


# %% ロジスティック回帰でうまく分類できない例

lr_miss = linear_model.LogisticRegression(random_state=0)
lr_miss.fit(X_std, y)

print(f"正解率: {lr_miss.score(X_std, y):.2f}")

plot_decision_regions(X_std, y, clf=lr_miss)


# %% hard-margin SVC

lsvc = svm.LinearSVC()
lsvc.fit(X_std, y)
plot_decision_regions(X_std, y, clf=lsvc)

# %% 線形分離できないデータ

X[19, 0] = 2
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

sc = preprocessing.StandardScaler()
sc.fit(X)
X_std = sc.transform(X)


# %% soft-margin SVC

lsvc = svm.LinearSVC(C=1)  # Cはペナルティの重さ。デフォルト1
# lsvc = svm.LinearSVC(C=0.2)
lsvc.fit(X_std, y)
plot_decision_regions(X_std, y, clf=lsvc)

# %% 決定木

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, random_state=41
)

# 決定木の適用（木の深さの制限なし,分割基準をジニ不純度に設定）
tree = DecisionTreeClassifier(max_depth=None, criterion="gini", random_state=41)
tree.fit(X_train, y_train)

# Accuracyの表示
print(f"訓練データの正解率:{tree.score(X_train,y_train):.3f}")
print(f"検証データの正解率:{tree.score(X_test,y_test):.3f}")

# 必要なライブラリのインポート
import graphviz
from sklearn.tree import export_graphviz

# Graphviz形式で決定木をエクスポートdot_data=export_graphviz(tree,out_file=None,impurity=False,filled=True,feature_names=wine.feature_names,class_names=wine.target_names)#Graphviz形式の決定木を表示graph=graphviz.Source(dot_data)graph


# %%
# %%
# %%
# %%
# %%

# print(f'{}')
