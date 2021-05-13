# adaboost实现鸢尾花分类

# 导入库
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# 导入鸢尾花数据
iris = load_iris()

# 数据集由csv格式保存
fea = iris.data   # 4个特征：花萼长度,花萼宽度,花瓣长度,花瓣宽度
lab = iris.target    # 3种类别：山鸢尾(Iris Setosa)，变色鸢尾(Iris Versicolor)，维吉尼亚鸢尾(Iris Virginica)
label_list = ['Iris Setosa', 'Iris Versicolor']

print(fea)
print(lab)

# 取前两项特征，前两个类别
fea = fea[lab < 2, :2]
lab = lab[lab < 2]


# 决策边界
def decision_regions(X, y, classifier=None):
    marker_list = ['o', 'x', 's']
    color_list = ['r', 'b', 'g']
    cmap = ListedColormap(color_list[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    t1 = np.linspace(x1_min, x1_max, 666)
    t2 = np.linspace(x2_min, x2_max, 666)

    x1, x2 = np.meshgrid(t1, t2)
    y_hat = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
    y_hat = y_hat.reshape(x1.shape)
    plt.contourf(x1, x2, y_hat, alpha=0.2, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for ind, clas in enumerate(np.unique(y)):
        plt.scatter(X[y == clas, 0], X[y == clas, 1], alpha=0.8, s=50, label=label_list[clas])


# 使用AdaBoostClassifier分类器
adbt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_split=30, min_samples_leaf=5),
                          algorithm="SAMME", n_estimators=10, learning_rate=0.3)

# 训练模型
adbt.fit(fea, lab)

# 设置AdaBoostClassifier分类器和超参数
AdaBoostClassifier(algorithm='SAMME',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=20,
            min_weight_fraction_leaf=0.0,  random_state=None,
            splitter='best'),
          learning_rate=0.8, n_estimators=10, random_state=None)

# 对分类结果进行可视化
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 设置正常显示符号
decision_regions(fea, lab, classifier=adbt)
plt.xlabel('Length（cm）')
plt.ylabel('Width（cm）')
plt.title('AdaBoost (n_e=10, l_r=0.3)',fontsize=20)
plt.legend()    # 添加标签
plt.show()
