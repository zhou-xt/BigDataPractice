import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if __name__ == '__main__':
    train_data = pd.read_csv('WoeData.csv')
    test_data = pd.read_csv('TestWoeData.csv')
    Y_train = train_data['SeriousDlqin2yrs']
    X_train = train_data.drop(['SeriousDlqin2yrs'], axis=1)
    Y_test = test_data['SeriousDlqin2yrs']
    X_test = test_data.drop(['SeriousDlqin2yrs'], axis=1)

    print("利用逻辑回归模型来做训练")
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    print("训练数据上的准确率为：%0.4f" % (model.score(X_train, Y_train)))
    print("测试数据上的准确率为：%0.4f" % (model.score(X_test, Y_test)))
    predicted = model.predict(X_test)
    print("精准率: ", precision_score(Y_test, predicted))
    print("召回率: ", recall_score(Y_test, predicted))
    print("F1: ", f1_score(Y_test, predicted))
    sn.heatmap(confusion_matrix(Y_test, predicted), annot=True)
    plt.show()

    print("利用决策树模型来做训练")
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    print("训练数据上的准确率为：%0.4f" % (model.score(X_train, Y_train)))
    print("测试数据上的准确率为：%0.4f" % (model.score(X_test, Y_test)))
    predicted = model.predict(X_test)
    print("精准率: ", precision_score(Y_test, predicted))
    print("召回率: ", recall_score(Y_test, predicted))
    print("F1: ", f1_score(Y_test, predicted))
    sn.heatmap(confusion_matrix(Y_test, predicted), annot=True)
    plt.show()

    print("利用KNN模型来做训练")
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    print("训练数据上的准确率为：%0.4f" % (model.score(X_train, Y_train)))
    print("测试数据上的准确率为：%0.4f" % (model.score(X_test, Y_test)))
    predicted = model.predict(X_test)
    print("精准率: ", precision_score(Y_test, predicted))
    print("召回率: ", recall_score(Y_test, predicted))
    print("F1: ", f1_score(Y_test, predicted))
    sn.heatmap(confusion_matrix(Y_test, predicted), annot=True)
    plt.show()

    print("利用支持向量机模型来做训练")
    model = SVC()
    model.fit(X_train, Y_train)
    print("训练数据上的准确率为：%0.4f" % (model.score(X_train, Y_train)))
    print("测试数据上的准确率为：%0.4f" % (model.score(X_test, Y_test)))
    predicted = model.predict(X_test)
    print("精准率: ", precision_score(Y_test, predicted))
    print("召回率: ", recall_score(Y_test, predicted))
    print("F1: ", f1_score(Y_test, predicted))
    sn.heatmap(confusion_matrix(Y_test, predicted), annot=True)
    plt.show()

    LogisticModel = LogisticRegression()
    LogisticModel.fit(X_train, Y_train)
    DecisionTreeModel = DecisionTreeClassifier()
    DecisionTreeModel.fit(X_train, Y_train)
    KNNModel = KNeighborsClassifier()
    KNNModel.fit(X_train, Y_train)
    SVMModel = SVC()
    SVMModel.fit(X_train, Y_train)

    logistic_disp = plot_roc_curve(LogisticModel, X_test, Y_test)
    model_disp = plot_roc_curve(DecisionTreeModel, X_test, Y_test, ax=logistic_disp.ax_)
    model_disp = plot_roc_curve(KNNModel, X_test, Y_test, ax=logistic_disp.ax_)
    model_disp = plot_roc_curve(SVMModel, X_test, Y_test, ax=logistic_disp.ax_)
    plt.show()
