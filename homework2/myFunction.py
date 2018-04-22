import copy
import pandas
from pandas import Series
import matplotlib
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
import numpy
import time
import math


# 项集数据结构
# m_item是['属性名1','属性名2','属性名3',...'属性名k']
# m_support是支持度
class Item:
    def __init__(self):
        self.m_item = [];
        self.m_support = -1;

    def __del__(self):
        self.m_item.clear();


# 生成频繁一项集
# 频繁k项应当是结构,包括(m_item, m_support),m_item['属性名1','属性名2','属性名3',...'属性名k']
# 频繁k项集应当是list of频繁k项
def GenerateSingleItemset(df, min_sup=1, attr_list=[]):
    if attr_list:
        attrList = attr_list;
    else:
        attrList = list(df.columns);
    currentFrequentItemset = list();
    infrequentItemset = list();
    for i in attr_list:
        currentItem = Item();
        currentItem.m_item.append(i);
        currentItem.m_support = (df[i] == 1).sum();
        if currentItem.m_support >= min_sup:
            currentFrequentItemset.append(currentItem);
        else:
            infrequentItemset.append(currentItem);
    return currentFrequentItemset, infrequentItemset;


def IsFrequentItem(item=Item(), infrequent_itemset=list()):
    if len(item.m_item) == 1:
        if item.m_item[-1] in infrequent_itemset:
            return False;
    else:
        tempItem = Item();
        tempItem.m_item.append(item.m_item[-2]);
        tempItem.m_item.append(item.m_item[-1]);
        for i in infrequent_itemset:
            if i.m_item == tempItem.m_item:
                return False;
    return True;


# 生成频繁k项集
# k_1_itemset里保存多个Item
def GenerateKItemset(df, min_sup=1, attr_list=[], k_1_itemset=[], infrequent_itemset=[]):
    if attr_list:
        attrList = attr_list;
    else:
        attrList = list(df.columns);
    currentFrequentItemset = list();
    # 利用两个k-1项生成一个k项
    for i in range(len(k_1_itemset)):
        priorItem = k_1_itemset[i];
        for j in range(i + 1, len(k_1_itemset)):
            postItem = k_1_itemset[j];
            # 打印显示
            # print('前项', priorItem.m_item, '支持度', priorItem.m_support);
            # print('后项', postItem.m_item, '支持度', postItem.m_support);
            # 判断是否满足拓展条件
            if (priorItem.m_item[0:-1] == postItem.m_item[0:-1] or priorItem.m_item[0] == postItem.m_item[0]) and \
                            priorItem.m_item[-1] in attrList and postItem.m_item[-1] in attrList and attrList.index(
                priorItem.m_item[-1]) < attrList.index(postItem.m_item[-1]):
                # 生成可能的k项
                newItem = copy.deepcopy(priorItem);
                newItem.m_item.append(postItem.m_item[-1]);
                # 再判断k项是否包含了非频繁的子项
                if IsFrequentItem(newItem, infrequent_itemset) == False:
                    continue;
                # 检验newItem中的属性组合是否达到出现次数要求
                newItem.m_support = ((df[newItem.m_item] == 1).sum(axis=1) == len(newItem.m_item)).sum();
                if newItem.m_support >= min_sup:
                    currentFrequentItemset.append(newItem);
                else:
                    infrequent_itemset.append(newItem);
            else:
                pass;
    return currentFrequentItemset, infrequent_itemset


# 把部分感兴趣的属性列作离散化
# nomi_attr中的Permit Type，Current Status，Existing Construction Type
# Permit Type有8种取值，Current Status有14种取值，Existing Construction Type有5种取值
# nume_attr中的Street Number
# 把Street Number按照四分位数划分为4组
# bina_attr中的全部属性
def ParticalDiscretization(df, nomi_attr=[], nume_attr=[], bina_attr=[]):
    new_df = copy.deepcopy(df[bina_attr]);
    for i in nomi_attr:
        # 针对标称属性i的不同取值，生成相应的新的二值属性列
        new_attr = [];
        for j in df[i].unique():
            new_attr.append(i + '=' + str(j));
        # 先生成一个临时的dataframe，根据df[]中原始属性列的取值填充临时dataframe，再把临时dataframe与new_df合并
        tmp = pandas.DataFrame(columns=new_attr, index=df.index);
        k = 0;
        for j in df[i]:
            tmp[i + '=' + str(j)][k] = 1;
            k = k + 1;
        # print(tmp);
        new_df = pandas.concat([new_df, tmp], axis=1);
    for i in nume_attr:
        # 针对数值属性列，用分位数划分为4个档次，生成新的二值属性
        new_attr = [i + '=0~0.25', i + '=0.25~0.5', i + '=0.5~0.75', i + '=0.75~1.0'];
        tmp = pandas.DataFrame(columns=new_attr, index=df.index);
        k = 0;
        for j in df[i]:
            if j >= df[i].quantile(0.75):
                tmp[i + '=0.75~1.0'][k] = 1;
            elif j >= df[i].quantile(0.5):
                tmp[i + '=0.5~0.75'][k] = 1;
            elif j >= df[i].quantile(0.25):
                tmp[i + '=0.25~0.5'][k] = 1;
            elif j >= df[i].quantile(0):
                tmp[i + '=0~0.25'][k] = 1;
            k = k + 1;
        new_df = pandas.concat([new_df, tmp], axis=1);
    new_df = new_df.fillna(value=0);
    return new_df;


def GenerateSubItem(item=[]):
    s = 1;
    total_sub_item = [];
    while s < (1 << len(item)) - 1:
        sub_item = [];
        for i in range(len(item)):
            if s & (1 << i):
                sub_item.append(item[i])
        total_sub_item.append(sub_item)
        s += 1
    return total_sub_item;
