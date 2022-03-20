import pandas as pd
import numpy as np
from pandas import Series
import scipy.stats.stats as stats
import matplotlib.pyplot as plt

# 定义自动分箱函数
def mono_bin(Y, X, n=20):
    r = 0
    good = Y.sum()
    bad = Y.count() - good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min'))
    print("=" * 60)
    print(d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe


# 自定义分箱函数
def self_bin(Y, X, cat):
    good = Y.sum()
    bad = Y.count() - good
    d1 = pd.DataFrame({'X': X, 'Y': Y, 'Bucket': pd.cut(X, cat)})
    d2 = d1.groupby('Bucket', as_index=True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min'))
    print("=" * 60)
    print(d4)
    woe = list(d4['woe'].round(3))
    return d4, iv, woe


# 用woe代替
def replace_woe(series, cut, woe):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(woe[m])
        i += 1
    return list


if __name__ == '__main__':
    data = pd.read_csv('TrainData.csv')
    pinf = float('inf')  # 正无穷大
    ninf = float('-inf')  # 负无穷大
    dfx1, ivx1, cutx1, woex1 = mono_bin(data.SeriousDlqin2yrs, data.age, n=10)
    dfx2, ivx2, cutx2, woex2 = mono_bin(data.SeriousDlqin2yrs, data.MonthlyIncome, n=10)
    # 连续变量离散化
    cutx3 = [ninf, 0, 1, 3, 5, pinf]
    cutx4 = [ninf, 0, 1, 2, 3, pinf]
    cutx5 = [ninf, 0, 1, 3, pinf]
    cutx6 = [ninf, 0, 1, 2, 3, 5, pinf]
    dfx3, ivx3, woex3 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
    dfx4, ivx4, woex4 = self_bin(data.SeriousDlqin2yrs, data['NumberRealEstateLoansOrLines'], cutx4)
    dfx5, ivx5, woex5 = self_bin(data.SeriousDlqin2yrs, data['NumberOfTime60-89DaysPastDueNotWorse'], cutx5)
    dfx6, ivx6, woex6 = self_bin(data.SeriousDlqin2yrs, data['NumberOfDependents'], cutx6)
    ivlist = np.array([ivx1, ivx3, ivx2, ivx4, ivx5, ivx6])
    index = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', ]
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(index)) + 1
    ax1.bar(x, ivlist, width=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(index, rotation=0, fontsize=12)
    ax1.set_ylabel('IV(Information Value)', fontsize=14)
    for a, b in zip(x, ivlist):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    plt.show()

    # 替换成woe
    data['age'] = Series(replace_woe(data['age'], cutx1, woex1))
    data['NumberOfTime30-59DaysPastDueNotWorse'] = Series(
        replace_woe(data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    data['MonthlyIncome'] = Series(replace_woe(data['MonthlyIncome'], cutx2, woex2))
    data['NumberRealEstateLoansOrLines'] = Series(replace_woe(data['NumberRealEstateLoansOrLines'], cutx4, woex4))
    data['NumberOfTime60-89DaysPastDueNotWorse'] = Series(
        replace_woe(data['NumberOfTime60-89DaysPastDueNotWorse'], cutx5, woex5))
    data['NumberOfDependents'] = Series(replace_woe(data['NumberOfDependents'], cutx6, woex6))
    data.to_csv('WoeData.csv', index=False)

    test = pd.read_csv('TestData.csv')
    test['age'] = Series(replace_woe(test['age'], cutx1, woex1))
    test['NumberOfTime30-59DaysPastDueNotWorse'] = Series(
        replace_woe(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
    test['MonthlyIncome'] = Series(replace_woe(test['MonthlyIncome'], cutx2, woex2))
    test['NumberRealEstateLoansOrLines'] = Series(replace_woe(test['NumberRealEstateLoansOrLines'], cutx4, woex4))
    test['NumberOfTime60-89DaysPastDueNotWorse'] = Series(
        replace_woe(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx5, woex5))
    test['NumberOfDependents'] = Series(replace_woe(test['NumberOfDependents'], cutx6, woex6))
    test.to_csv('TestWoeData.csv', index=False)
