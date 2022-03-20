import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv("cs-training.csv")

# 去除重复值
df.duplicated()
df.drop_duplicates()
# 为了防止引入噪声，将有缺失值的行均丢弃
df = df.dropna()
# 异常值处理，对年龄等于0的异常值进行剔除
df = df[df['age'] > 0]
df = df[df['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
# 删除不分析的列，特征选择
columns = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "NumberOfOpenCreditLinesAndLoans",
           "NumberOfTimes90DaysLate"]
df.drop(columns, axis=1, inplace=True)
# 保存处理后的文件
df.to_csv("data.csv")

# 训练集和测试集的切分
data = pd.read_csv('data.csv')
Y = data['SeriousDlqin2yrs']
x = [0, 1, 2]
X = data.drop(data.columns[x], axis=1)
# 测试集占比30%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
train = pd.concat([Y_train, X_train], axis=1)
test = pd.concat([Y_test, X_test], axis=1)
clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
train.to_csv('TrainData.csv', index=False)
test.to_csv('TestData.csv', index=False)
