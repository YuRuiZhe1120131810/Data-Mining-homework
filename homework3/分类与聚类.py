from myFunction import *

##########
# 分类预测
##########
TrainSet = pandas.read_csv(open('D:\数据挖掘作业\数据挖掘作业三\训练集.csv'))
TestSet = pandas.read_csv(open('D:\数据挖掘作业\数据挖掘作业三\测试集.csv'))

RedundantAttribute = ['PassengerId', 'Name']
NominalAttribute = ['Pclass', 'Sex', 'Cabin', 'Embarked']
NumericAttribute = ['SibSp', 'Age', 'Parch', 'Fare']

# 丢弃有空值行
TrainSet = TrainSet.dropna(axis=0, how='any')

# Sex属性male转为1，female转为0
TrainSet.replace(to_replace='male', value=1, inplace=True)
TrainSet.replace(to_replace='female', value=0, inplace=True)
TestSet.replace(to_replace='male', value=1, inplace=True)
TestSet.replace(to_replace='female', value=0, inplace=True)

# Age属性nan填充均值
TrainSet['Age'] = TrainSet['Age'].fillna(value=TrainSet['Age'].mean())
TestSet['Age'] = TestSet['Age'].fillna(value=TestSet['Age'].mean())

# Fare属性nan填充均值
TestSet['Fare'] = TestSet['Fare'].fillna(value=TestSet['Fare'].mean())

# 把字符型标称属性映射为数值型
to_encode_attr = ['Embarked', 'Cabin', 'Ticket'];
TrainSet = encode_target(TrainSet, to_encode_attr)
TestSet = encode_target(TestSet, to_encode_attr)

# 把标签修改成true false
# TrainSet['Survived'] = TrainSet['Survived'].replace(to_replace=1, value=True)
# TrainSet['Survived'] = TrainSet['Survived'].replace(to_replace=0, value=False)

# 把Age与Fare修改为整数
TrainSet[['Age', 'Fare']] = TrainSet[['Age', 'Fare']].astype(int)
TestSet[['Age', 'Fare']] = TestSet[['Age', 'Fare']].astype(int)

# 丢弃多余属性列
TrainSet = TrainSet.drop(columns=RedundantAttribute)
TestSet = TestSet.drop(columns=RedundantAttribute)

# 特征属性与标签属性
LabelAttr = ['Survived']
FeatureAttr = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket'];
Label = TrainSet[LabelAttr]
Feature = TrainSet[FeatureAttr]

# 定义决策树并训练
# DesicionTree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
DesicionTree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
DesicionTree.fit(Feature, Label)

# 使用决策树分类
result1 = DesicionTree.predict(TestSet)
# print(result1)

# 定义高斯分布的朴素贝叶斯分类器
GaussianNaiveBayesClassifier = GaussianNB().fit(Feature, Label)

# 使用高斯朴素贝叶斯分类器分类
result2 = GaussianNaiveBayesClassifier.predict(TestSet)
# print(result2)

# 比较两种分类器的分类结果
if len(result1) == len(result2):
    print('总人数:', Series(result2).count(), '决策树判断生还者:', Series(result1).sum(), '贝叶斯判断生还者:', Series(result2).sum(),
          '两者的相同预测数量:',
          (Series(result1) == Series(result2)).sum())

# 显示分类的结果，横坐标为年龄、纵坐标为票价、红点为死亡、绿点为存活
# 先显示决策树的预测结果
pyplot.figure()

temp = pandas.concat([DataFrame(result1, columns=['Survived']), TestSet[['Age', 'Fare']]], axis=1)
alive = temp.loc[temp['Survived'] == 1]
dead = temp.loc[temp['Survived'] == 0]

figure1 = pyplot.subplot(2, 1, 1)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure1.set_title(u'决策树预测结果')
alive_distribute = figure1.scatter(alive['Age'], alive['Fare'], c='green', marker='d')
dead_distribute = figure1.scatter(dead['Age'], dead['Fare'], c='red', marker='*')
pyplot.xlabel(u'生还者或遇难者年龄')
pyplot.ylabel(u'生还者或遇难者票价')
figure1.legend((alive_distribute, dead_distribute), ('生还者', '遇难者'), loc=1)

temp = pandas.concat([DataFrame(result2, columns=['Survived']), TestSet[['Age', 'Fare']]], axis=1)
alive = temp.loc[temp['Survived'] == 1]
dead = temp.loc[temp['Survived'] == 0]

figure2 = pyplot.subplot(2, 1, 2)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure2.set_title(u'高斯贝叶斯预测结果')
alive_distribute = figure2.scatter(alive['Age'], alive['Fare'], c='green', marker='d')
dead_distribute = figure2.scatter(dead['Age'], dead['Fare'], c='red', marker='*')
pyplot.xlabel(u'生还者或遇难者年龄')
pyplot.ylabel(u'生还者或遇难者票价')
figure2.legend((alive_distribute, dead_distribute), ('生还者', '遇难者'), loc=1)

pyplot.show()

###########
# 聚类分析 #
###########
# 重新载入内容
TrainSet = pandas.read_csv(open('D:\数据挖掘作业\数据挖掘作业三\训练集.csv'))
TestSet = pandas.read_csv(open('D:\数据挖掘作业\数据挖掘作业三\测试集.csv'))

RedundantAttribute = ['PassengerId', 'Name', 'Ticket']
NominalAttribute = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Ticket']
NumericAttribute = ['SibSp', 'Age', 'Parch', 'Fare']

# 把属性中的字符串值改为数值
to_encode_attr = ['Embarked', 'Cabin', 'Ticket'];
TrainSet = encode_target(TrainSet, to_encode_attr)

# Sex属性male转为1，female转为0
TrainSet['Sex'].replace(to_replace='male', value=1, inplace=True)
TrainSet['Sex'].replace(to_replace='female', value=0, inplace=True)
TestSet['Sex'].replace(to_replace='male', value=1, inplace=True)
TestSet['Sex'].replace(to_replace='female', value=0, inplace=True)

# Age属性nan填充均值
TrainSet['Age'] = TrainSet['Age'].fillna(value=TrainSet['Age'].mean())
TestSet['Age'] = TestSet['Age'].fillna(value=TestSet['Age'].mean())

# Fare属性nan填充均值
TestSet['Fare'] = TestSet['Fare'].fillna(value=TestSet['Fare'].mean())

# 丢弃多余属性列
TrainSet = TrainSet.drop(columns=RedundantAttribute)
TestSet = TestSet.drop(columns=RedundantAttribute)

# k均值聚类
kmeans_cluster = KMeans(n_clusters=4, random_state=0).fit(TrainSet)

# DBSCAN聚类
dbscan_cluster = DBSCAN(eps=10, min_samples=5).fit(TrainSet)

# 显示聚类结果
pyplot.figure()

# kmeans聚类结果显示
temp = pandas.concat([DataFrame(kmeans_cluster.labels_, columns=['ClassLabel']), TrainSet], axis=1)
type1 = temp.loc[temp['ClassLabel'] == 0]
type2 = temp.loc[temp['ClassLabel'] == 1]
type3 = temp.loc[temp['ClassLabel'] == 2]
type4 = temp.loc[temp['ClassLabel'] == 3]

figure1 = pyplot.subplot(2, 1, 1)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure1.set_title(u'k-means聚类结果')

type1_distribute = figure1.scatter(type1['Age'], type1['Fare'], c='green', marker='d')
type2_distribute = figure1.scatter(type2['Age'], type2['Fare'], c='red', marker='*')
type3_distribute = figure1.scatter(type3['Age'], type3['Fare'], c='black', marker='p')
type4_distribute = figure1.scatter(type4['Age'], type4['Fare'], c='brown')
pyplot.xlabel(u'某类人的年龄')
pyplot.ylabel(u'某类人的票价')
figure1.legend((type1_distribute, type2_distribute, type3_distribute, type4_distribute), ('第一类', '第二类', '第三类', '第四类'),
               loc=1)

# DBSCAN聚类结果显示
temp = pandas.concat([DataFrame(dbscan_cluster.labels_, columns=['ClassLabel']), TrainSet], axis=1)
temp = temp.drop(temp[temp['ClassLabel'] == -1].index)
print('DBSCAN聚类个数:', len(temp['ClassLabel'].unique()), 'DBSCAN类标', temp['ClassLabel'].unique())
type1 = temp.loc[temp['ClassLabel'] == 0]
type2 = temp.loc[temp['ClassLabel'] == 1]
type3 = temp.loc[temp['ClassLabel'] == 2]
type4 = temp.loc[temp['ClassLabel'] == 3]

figure2 = pyplot.subplot(2, 1, 2)
pyplot.rcParams['font.sans-serif'] = ['SimHei']
figure2.set_title(u'DBSCAN聚类结果')

type1_distribute = figure2.scatter(type1['Age'], type1['Fare'], c='green', marker='d')
type2_distribute = figure2.scatter(type2['Age'], type2['Fare'], c='red', marker='*')
type3_distribute = figure2.scatter(type3['Age'], type3['Fare'], c='black', marker='p')
type4_distribute = figure2.scatter(type4['Age'], type4['Fare'], c='brown')
pyplot.xlabel(u'某类人的年龄')
pyplot.ylabel(u'某类人的票价')
figure2.legend((type1_distribute, type2_distribute, type3_distribute, type4_distribute), ('第一类', '第二类', '第三类', '第四类'),
               loc=1)

pyplot.show()
