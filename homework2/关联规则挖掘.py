# coding = utf-8
from myFunction import *

DataFile = open("D:\数据挖掘作业\数据挖掘作业二\Building_Permits.csv")
DataTable = pandas.read_csv(DataFile);
# 剔除全空值的列
DataTable = DataTable.dropna(axis=1, how='all');
# 标称属性
NominalAttribute = ['Permit Number', 'Permit Type', 'Permit Type Definition', 'Permit Creation Date', 'Block', 'Lot',
                    'Street Number', 'Street Number Suffix', 'Street Name', 'Street Name Suffix', 'Unit suffix',
                    'Description', 'Current Status', 'Current Status Date', 'Filed Date', 'Issued Date',
                    'Completed Date', 'First Construction Document Date', 'Structural Notification', 'Fire Only Permit',
                    'Permit Expiration Date', 'Existing Use', 'Proposed Use', 'Plansets', 'TIDF Compliance',
                    'Existing Construction Type', 'Existing Construction Type Description',
                    'Proposed Construction Type', 'Proposed Construction Type Description', 'Site Permit',
                    'Neighborhoods - Analysis Boundaries', 'Zipcode', 'Location'];
# 标称属性摘要，是字典的字典，外层字典的键是属性列名，内层字典的键是各属性的取值
NominalAttributeAbstract = dict();

# 二值属性：
BinaryAttribute = ['Structural Notification', 'Voluntary Soft-Story Retrofit', 'Fire Only Permit', 'Site Permit'];

# 数值属性
NumericAttribute = ['Unit', 'Number of Existing Stories', 'Number of Proposed Stories', 'Estimated Cost',
                    'Revised Cost', 'Existing Units', 'Proposed Units', 'Supervisor District'];
# 数值属性摘要，是字典，键是属性列名，值是list[7]，依次表示最大、最小、均值、中位数、四分之一位数、四分之三位数、缺失值个数
NumericAttributeAbstract = dict();

# 冗余属性
# Completed Date会与Current Status Date相同、Record ID是无用属性、TIDF Compliance只有两个有效元组
RedundantAttribute = ['Completed Date', 'Record ID', 'TIDF Compliance'];

# 丢弃冗余属性
DataTable = DataTable.drop(columns=RedundantAttribute);

# 对于二值属性，可以把空值填充为N
DataTable[BinaryAttribute] = DataTable[BinaryAttribute].fillna(value=0);
DataTable[BinaryAttribute] = DataTable[BinaryAttribute].replace(to_replace='Y', value=1)
DataTable[BinaryAttribute] = DataTable[BinaryAttribute].replace(to_replace='N', value=0)

# 把Unit属性的空值填充为0
DataTable[['Unit']] = DataTable[['Unit']].fillna(value=0);

# 把Supervisor District属性的空值填充为0
DataTable[['Supervisor District']] = DataTable[['Supervisor District']].fillna(value=0);

# Apriori算法只能处理二值属性列，需要把标称属性列、数值属性列给离散化，转换为二值属性
# 把部分感兴趣的属性列作离散化
# 标称的Permit Type，Current Status，Existing Construction Type
# Permit Type有8种取值，Current Status有14种取值，Existing Construction Type有5种取值
# 数值的Street Number
# 把Street Number按照四分位数划分为4组
if os.path.exists('D:\数据挖掘作业\数据挖掘作业二\二值属性.CSV') == False:
    DataTable = ParticalDiscretization(DataTable, ['Permit Type', 'Current Status', 'Existing Construction Type'],
                                       ['Street Number'], BinaryAttribute);
    DataTable.to_csv(path_or_buf='D:\数据挖掘作业\数据挖掘作业二\二值属性.CSV');
else:
    DataTable = pandas.read_csv(open('D:\数据挖掘作业\数据挖掘作业二\二值属性.CSV'));

# 获取当前可用的列
CurrentAvailableAttribute = list(DataTable.columns);

# 项集的最小支持度
MinimalSupport = 8000;

# 所有频繁项集的集合，第一个元素是频繁一项集，第二个元素是频繁二项集，第三个元素是频繁三项集，依次类推
frequentItemsetSet = [];

# 非频繁项集，杂糅包含各个非频繁的项
infrequentItemset = [];

for i in range(len(CurrentAvailableAttribute)):
    # 最长的项包含所有属性列
    if frequentItemsetSet:
        # FrequentItemsetSet非空，生成频繁k项集
        lastItemset = frequentItemsetSet[-1];
        currentItemset, infrequentItemset = GenerateKItemset(DataTable, MinimalSupport, CurrentAvailableAttribute,
                                                             lastItemset, infrequentItemset);
        if currentItemset:
            frequentItemsetSet.append(currentItemset)
        else:
            print('没有更长的频繁项模式了，频繁模式挖掘完毕');
            break;
    else:
        # FrequentItemsetSet为空，生成频繁一项集
        currentItemset, infrequentItemset = GenerateSingleItemset(DataTable, MinimalSupport, CurrentAvailableAttribute);
        frequentItemsetSet.append(currentItemset)
        pass;

# 项集的最小置信度
MinimalConfidence = 0.9;

for i in frequentItemsetSet:
    print('频繁', len(i[0].m_item), '项:');
    for j in i:
        SubItem = GenerateSubItem(j.m_item);
        for antecedent in SubItem:  # 把子项当前件，把子项的补当后件
            consequent = list(set(j.m_item) ^ set(antecedent));
            # print('前件:', antecedent, '后件:', consequent);
            consequent_support = ((DataTable[consequent] == 1).sum(axis=1) == len(
                consequent)).sum();  # 后件的支持度
            antecedent_support = ((DataTable[antecedent] == 1).sum(axis=1) == len(
                antecedent)).sum();  # 前件的支持度
            confidence = j.m_support / consequent_support;
            if confidence > MinimalConfidence:
                lift = j.m_support / consequent_support;
                print(antecedent, '=>', consequent, '置信度:', confidence, '提升度:', lift);
