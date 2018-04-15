# coding = utf-8
import pandas
from pandas import Series
import matplotlib
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
import numpy
import time
import math

DataFile = open("D:\数据挖掘作业一\\NFL Play by Play 2009-2017 (v4).csv")
DataTable = pandas.read_csv(DataFile);
# 剔除全空的列
DataTable = DataTable.dropna(axis=1, how='all');
# 剔除属性值全是一样的列
uniques = DataTable.nunique();
DataTable = DataTable.drop(uniques[uniques == 1].index, axis=1);
# 标称属性
NominalAttribute = ['Date', 'GameID', 'time', 'SideofField', 'FirstDown', 'posteam', 'DefensiveTeam', 'desc',
                    'PlayAttempted', 'Yards.Gained', 'sp', 'Touchdown', 'ExPointResult', 'TwoPointConv', 'DefTwoPoint',
                    'Onsidekick', 'Safety', 'PuntResult', 'PlayType', 'Passer', 'Passer_ID', 'PassAttempt',
                    'PassOutcome', 'PassLength', 'QBHit', 'PassLocation', 'InterceptionThrown', 'Interceptor', 'Rusher',
                    'Rusher_ID', 'RushAttempt', 'RunLocation', 'RunGap', 'Receiver', 'Receiver_ID', 'Reception',
                    'ReturnResult', 'Returner', 'BlockingPlayer', 'Tackler1', 'Tackler2', 'FieldGoalResult', 'Fumble',
                    'RecFumbTeam', 'RecFumbPlayer', 'Sack', 'Challenge.Replay', 'ChalReplayResult', 'Accepted.Penalty',
                    'PenalizedTeam', 'PenaltyType', 'PenalizedPlayer', 'HomeTeam', 'AwayTeam', 'Timeout_Indicator',
                    'Timeout_Team', 'Season', 'posteam_timeouts_pre', 'HomeTimeouts_Remaining_Pre',
                    'AwayTimeouts_Remaining_Pre', 'HomeTimeouts_Remaining_Post', 'AwayTimeouts_Remaining_Post'];
# 标称属性摘要，是字典的字典，外层字典的键是属性列名，内层字典的键是各属性的取值
NominalAttributeAbstract = dict();
# 数值属性
NumericAttribute = ['Drive', 'qtr', 'down', 'TimeUnder', 'TimeSecs', 'PlayTimeDiff', 'yrdln', 'yrdline100', 'ydstogo',
                    'ydsnet', 'GoalToGo', 'AirYards', 'YardsAfterCatch', 'FieldGoalDistance', 'Penalty.Yards',
                    'PosTeamScore', 'DefTeamScore', 'ScoreDiff', 'AbsScoreDiff', 'No_Score_Prob', 'Opp_Field_Goal_Prob',
                    'Opp_Safety_Prob', 'Opp_Touchdown_Prob', 'Field_Goal_Prob', 'Safety_Prob', 'Touchdown_Prob',
                    'ExPoint_Prob', 'TwoPoint_Prob', 'ExpPts', 'EPA', 'airEPA', 'yacEPA', 'Home_WP_pre', 'Away_WP_pre',
                    'Home_WP_post', 'Away_WP_post', 'Win_Prob', 'WPA', 'airWPA', 'yacWPA'];
# 数值属性摘要，是字典，键是属性列名，值是list[7]，依次表示最大、最小、均值、中位数、四分之一位数、四分之三位数、缺失值个数
NumericAttributeAbstract = dict();
# 二值属性
BinaryAttribute = ['GoalToGo', 'FirstDown', 'sp', 'Touchdown', 'ExPointResult', 'Onsidekick', 'PuntResult',
                   'PassAttempt', 'PassOutcome', 'PassLength', 'QBHit', 'InterceptionThrown', 'RushAttempt',
                   'Reception', 'Fumble', 'Sack', 'Challenge.Replay', 'ChalReplayResult', 'Accepted.Penalty',
                   'Timeout_Indicator'];

# 统计数据摘要
for i in DataTable.columns:  # 针对每一个属性列
    if i in NominalAttribute:  # 如果是标称属性
        DataColumn = DataTable[i];  # 获取该列
        DataColumnStatistic = DataColumn.value_counts();  # 计算一列上每个取值的频数
        tmpDict = DataColumnStatistic.to_dict();  # 取值为键，频数为值
        NominalAttributeAbstract[i] = tmpDict;
        print('属性', i, '的每个取值的频数');
        print(tmpDict);
    elif i in NumericAttribute:  # 如果是数值属性
        DataColumn = DataTable[i];  # 获取该列
        tmpList = [DataColumn.max(), DataColumn.min(), DataColumn.mean(), DataColumn.median(),
                   DataColumn.quantile(0.25), DataColumn.quantile(0.75), DataColumn.isnull().sum()];
        NumericAttributeAbstract[i] = tmpList;
        print('属性', i, '的最大值=', tmpList[0], '最小值=', tmpList[1], '平均值=', tmpList[2], '中值', tmpList[3], '四分之一位数=',
              tmpList[4], '四分之三位数=', tmpList[5], '空值数=', tmpList[6]);
    else:
        print('属性', i, '既不是标签也不是数值属性');
else:
    print('数据摘要统计完成');

# 针对数值属性：绘制直方图
matplotlib.rcParams['font.sans-serif'] = ['SimHei'];
AvaiableNumericAttribute = [];
for i in DataTable.columns:
    if i in NumericAttribute:
        AvaiableNumericAttribute.append(i);
DataTable.hist(column=AvaiableNumericAttribute);
pyplot.show();
# pyplot.draw()
# pyplot.pause(0.1)
# pyplot.close();
print('直方图绘制完成');

# 针对数值属性：绘分位数图
for i in DataTable.columns:
    if i in NumericAttribute:
        DataColumn = DataTable[i];  # 获取该列
        QuantileSequence = DataColumn.quantile(numpy.arange(0, 1, 0.01));  # 获取0%到100%的分位数
        QuantileSequence.plot(title='属性' + i + '分位数图');  # 绘制数据的分位数图
        GaussianDistribution = Series(numpy.random.normal(loc=DataColumn.mean(), scale=numpy.sqrt(DataColumn.var()),
                                                          size=1000));  # 以均值和标准差生成1000个高斯样本
        GaussianDistribution.quantile(numpy.arange(0, 1, 0.01)).plot();  # 绘制高斯样本的分位数图
        pyplot.show();
        # pyplot.draw()
        # pyplot.pause(0.1)
        # pyplot.close();
else:
    print('分位数图绘制完成');

# 针对数值属性：绘制盒图
DataTable.boxplot(column=NumericAttribute);
pyplot.xlabel('各属性列');
pyplot.ylabel('离群点与盒图');
pyplot.show();
# pyplot.draw();
# pyplot.pause(0.01)
# print(list(DataTable.columns.values));#查看所有的列
# print(DataTable.iloc[1])#查看某行

# 处理缺失值：将缺失部分剔除
temp = DataTable.copy(deep=True);
temp = temp.dropna(axis=0, how='any');  # 有空属性则剔除元组
print('丢弃缺失值元祖');

# 处理缺失值：用众数填补缺失值
temp = DataTable.copy(deep=True);
for i in temp.columns:
    DataColumn = temp[i];  # 获取该列
    MostFrequentElement = DataColumn.value_counts().idxmax();
    # print('属性', i, '的众数是', str(MostFrequentElement));
    DataColumn = DataColumn.fillna(value=MostFrequentElement);  # 众数填补缺失值
    temp[i] = DataColumn;
else:
    print('缺失值用众数填补完成');

# 处理缺失值：针对数值属性，用相关性填补缺失值
# 针对某一个元祖，其数值属性上有空值，首先查看其他数值属性上的非空项，搜索出最相关的属性列，然后计算线性回归系数，由最相关属性值推算出缺失属性值
temp = DataTable.copy(deep=True);
CorrMat = temp.corr();
for i in temp.columns:
    pass;
else:
    pass;


def similarity(row1, row2, NominalAttr, NumericAttr):
    nominal_attr_simi = 0;
    for i in NominalAttr:
        if i in row1.index and i in row2.index and row1.loc[i] == row2.loc[i] and row1.loc[i] != numpy.nan and row2.loc[
            i] != numpy.nan:
            nominal_attr_simi = nominal_attr_simi + 1;
    nominal_attr_simi = nominal_attr_simi;
    numeric_attr_simi = 0;
    for i in NumericAttr:
        if numpy.isnan(row1.loc[i]).all() == False and numpy.isnan(row2.loc[i]).all() == False and (
            i in row1.index) and (i in row2.index):
            numeric_attr_simi = numeric_attr_simi + (row1.loc[i] - row2.loc[i]) * (row1.loc[i] - row2.loc[i]);
    numeric_attr_simi = math.sqrt(numeric_attr_simi);
    return (nominal_attr_simi, numeric_attr_simi);


# 处理缺失值：通过行的相似性填补缺失值
# 首先计算元组两两之间的相似性，针对某个有缺失值的元祖a，找出与其最相似的且相应字段上非空的元组b，用b的字段替换a的空字段
temp = DataTable.copy(deep=True);
total_rows = len(temp.index);
for i in range(0, total_rows):
    if temp.iloc[i].isnull().sum() > 0:  # 该元组有属性缺失
        best_close = i;
        best_close_measure = (0, float("inf"));
        for j in range(0, total_rows):
            if j != i:  # and temp.iloc[j].isnull().any()==False:
                similarity_measure = similarity(temp.iloc[i], temp.iloc[j], NominalAttribute, NumericAttribute);
                if similarity_measure[0] > best_close_measure[0] and similarity_measure[1] < best_close_measure[1]:
                    best_close = j;
                    best_close_measure = similarity_measure;
        if best_close == i and best_close_measure == (0, float("inf")):
            print('没有找到合适的近似元组');
        for k in range(0, temp.columns.size):
            if str(temp.iat[i, k]) == 'nan':
                temp.iat[i, k] = temp.iat[j, k];
else:
    if temp.equals(DataTable):
        print('没有填充缺失值');
    else:
        print(temp);
