# -*- coding: utf-8 -*-

from flask import Flask,render_template
import jieba
import pandas as pd
import json
from collections import Counter
import re

# 正面评价：1，中性评价是：0，负面评价是：-1
# 构建朴素贝叶斯分类器
class Bayesianclassifier:

    def __init__(self):
        self.training_set = []  # 给定训练集
        self.testing_set = []  # 给定测试集
        self.positive_comment = []  # 正面评价数集
        self.negative_comment = []  # 负面评价数集
        self.neutral_comment = []  # 中性评价数集
        self.train_table = []  # 给定训练集标签
        self.positive_por = 0  # 训练集正面评价概率
        self.negative_pro = 0  # 训练集负面评价概率
        self.neutral_pro = 0  # 训练中性评价概率
        self.positive_words_pro = {}  # 存放正面评价单词概率
        self.neutral_words_pro = {}  # 存放中性评价单词概率
        self.negative_words_pro = {}  # 存放中性评价单词概率
        self.pro = 1  # 初始化概率
        self.words_Total = 0  # 特征词个数
        self.predicted_table = []  # 预测标签
        self.positivewords_classnum = 0
        self.neutralwords_classnum = 0
        self.negativewords_classnum = 0

    # 拟合模型#a:平滑系数

    def Fitting_model(self, train_set, train_table, a):
        self.training_set = Bayesianclassifier.word_processing(train_set)
        self.train_table = train_table
        for i in range(len(self.train_table)):
            if self.train_table[i] == 1:
                self.positive_comment.append(self.training_set[i])
            elif self.train_table[i] == 0:
                self.neutral_comment.append(self.training_set[i])
            elif self.train_table[i] == -1:
                self.negative_comment.append(self.training_set[i])
        self.positive_por = len(self.positive_comment) / len(self.training_set)  # 训练集正面评价概率
        self.negative_pro = len(self.negative_comment) / len(self.training_set)  # 训练集负面评价概率
        self.neutral_pro = len(self.neutral_comment) / len(self.training_set)  # 训练中性评价概率
        words_count1 = {}
        words = []
        # 求该训练集特征单词个数words_Tptal
        for i in self.training_set:
            for j in i:
                if j not in words:
                    self.words_Total += 1
                    words.append(j)
                else:
                    self.words_Total = self.words_Total
        # 正面分类中所有单词的总数positivewords_classnum
        for i in self.positive_comment:
            self.positivewords_classnum += len(i)

        # 中性分类中所有单词的总数neutralwords_classnum
        for i in self.neutral_comment:
            self.neutralwords_classnum += len(i)

        # 负面分类中所有单词的总数negativewords_classnum
        for i in self.negative_comment:
            self.negativewords_classnum += len(i)

        # 求正面评价词典概率
        for i in self.positive_comment:
            for j in i:
                if j in words_count1.keys():
                    words_count1[j] += 1
                else:
                    words_count1[j] = 1
        for j in words_count1.keys():
            self.positive_words_pro[j] = Bayesianclassifier.Laplace(words_count1[j], self.positivewords_classnum,
                                                                    self.words_Total, a)

        # 求中性评价词典概率
        words_count2 = {}
        for i in self.neutral_comment:
            for j in i:
                if j in words_count2.keys():
                    words_count2[j] += 1
                else:
                    words_count2[j] = 1
        for j in words_count2.keys():
            self.neutral_words_pro[j] = Bayesianclassifier.Laplace(words_count2[j], self.neutralwords_classnum,
                                                                   self.words_Total, a)

        # 求负面评价词典概率
        words_count3 = {}
        for i in self.negative_comment:
            for j in i:
                if j in words_count3.keys():
                    words_count3[j] += 1
                else:
                    words_count3[j] = 1
        for j in words_count3.keys():
            self.negative_words_pro[j] = Bayesianclassifier.Laplace(words_count3[j], self.negativewords_classnum,
                                                                    self.words_Total, a)

    # 预测舆情
    # a:平滑系数
    def predicted(self, testing_set, a):
        testing_set = Bayesianclassifier.word_processing(testing_set)
        predicted = []
        result_predicted = {'predicted_psitive': "正面评论", 'predicted_neutral': "中性评论", 'predicted_negative': "负面评论"}
        pro = 1
        positive_words_pro_keys = [i for i in self.positive_words_pro.keys()]
        neutral_words_pro_keys = [i for i in self.neutral_words_pro.keys()]
        negative_words_pro_keys = [i for i in self.negative_words_pro.keys()]

        # 该数据集是正面评论的概率
        for i in testing_set:
            for j in i:
                key = j
                if key in positive_words_pro_keys:
                    pro = self.pro * self.positive_words_pro[key]
                else:
                    pro = pro * Bayesianclassifier.Laplace(0, self.positivewords_classnum, self.words_Total, a)
            predicted_psitive = pro * self.positive_por
            # 该数据集是中性评论的概率
            for j in i:
                key = j
                if key in neutral_words_pro_keys:
                    pro = self.pro * self.neutral_words_pro[key]
                else:
                    pro = pro * Bayesianclassifier.Laplace(0, self.neutralwords_classnum, self.words_Total, a)
            predicted_neutral = pro * self.neutral_pro
            # 该数据集是负面评论的概率
            for j in i:
                key = j
                if key in negative_words_pro_keys:
                    pro = self.pro * self.negative_words_pro[key]
                else:
                    pro = pro * Bayesianclassifier.Laplace(0, self.negativewords_classnum, self.words_Total, a)
            predicted_negative = pro * self.negative_pro
            # 找出最大后验数
            if predicted_psitive > predicted_negative and predicted_psitive > predicted_neutral:
                predicted_psitive = "predicted_psitive"
                predicted.append(result_predicted[predicted_psitive])
                self.predicted_table.append(1)
            elif predicted_negative > predicted_psitive and predicted_negative > predicted_neutral:
                predicted_negative = "predicted_negative"
                self.predicted_table.append(-1)
                predicted.append(result_predicted[predicted_negative])
            else:
                predicted_neutral = "predicted_neutral"
                self.predicted_table.append(0)
                predicted.append(result_predicted[predicted_neutral])
        # print("预测分类标签：", self.predicted_table)
        # print(len(self.predicted_table))
        for wordList in testing_set:
            for wordStr in wordList:
                wordStr = wordStr.replace('[0-9a-zA-Z]', '')

        return self.predicted_table


    # 语句分词
    @staticmethod
    def word_processing(wait_date):
        date = []
        for i in wait_date:
            date1 = jieba.lcut(i)
            date.append(date1)
        # print(date)
        return date

    # 拉普拉斯平滑系数
    # words_Number:该分类中某单词个数
    # words_classnum:该分类中所有单词的总数
    # words_Total:特征单词个数
    # a:平滑系数
    @staticmethod
    def Laplace(words_Number, words_classnum, words_Total, a):
        return (words_Number + a) / (words_classnum + a * words_Total)

    # 模型评估
    def evaluation_model(self, true_table):
        correct = 0
        error = 0
        for i in range(len(self.predicted_table)):
            if self.predicted_table[i] == true_table[i]:
                correct += 1
            else:
                error += 1
        recall = float(correct / (error + correct))  # 召回率
        precision = float(correct / len(self.predicted_table))  # 准确性
        strprecision = "模型召回率： " + str(recall)
        strrecall = "模型准确性： " + str(precision)
        return strprecision, strrecall

    # 统计预测数据
    def statistical_data(self):
        pos = 0  # 正面评论数
        neg = 0  # 负面评论数
        neu = 0  # 中性评论数
        for i in self.predicted_table:
            if i == 1:
                pos += 1
            elif i == 0:
                neu += 1
            else:
                neg += 1
        # expect = pos / len(self.predicted_table) * 1 - neg / len(self.predicted_table)
        # print('正面评论百分比: {:.2%}'.format(pos/len(self.predicted_table)))
        # print('中性评论百分比: {:.2%}'.format(neu/len(self.predicted_table)))
        # print('负面评论百分比: {:.2%}'.format(neg/len(self.predicted_table)))
        # print("舆论期望：",expect)
        expect = '舆情期望：' + str(pos / len(self.predicted_table) * 1 - neg / len(self.predicted_table))[:12]
        positive = pos / len(self.predicted_table)
        neutral = neu / len(self.predicted_table)
        negative = neg / len(self.predicted_table)
        return positive, neutral, negative, expect


def get_date(path):
    df_data = pd.read_excel('qzcomment.xlsx', header=None, skiprows=1)
    df = pd.read_excel('qzcomment.xlsx')
    # print(df_data)
    data_set = df_data.iloc[:, 0:1].values
    data_table = df_data.iloc[:, 1:2].values
    table_i = []
    [table_i.append(data_table[i][0]) for i in range(0, len(data_table))]
    lst = []
    for i in range(0, len(data_set)):
        str_i = data_set[i][0]
        lst.append(str_i)
    return table_i, lst,df

def cut_word(sentence):
    with open(r'mystopwords.txt', encoding='UTF-8') as words:
        stop_words = [i.strip() for i in words.readlines()]
    words = [i for i in jieba.lcut(sentence) if i not in stop_words]  #切词过程中删除停止词
    # 切完的词用空格隔开
    # result = ' '.join(words)
    return words

path = 'qzcomment.xlsx'  # set workingData
jieba.load_userdict(r'all_words.txt')
table_i,lst,df = get_date(path)
train_set = lst[:900] #
train_table = table_i[:900]
test_set = lst[900:]
test_table = table_i[900:]
df.saying = df.saying.str.replace(r"[0-9a-zA-Z楼. \u3000口口ۖ': 2741, 'ิ'ۣ']",'')
pattern = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
words = df.saying.apply(cut_word)
wordsListAll = []
for wordsList in words:
    for word in wordsList:
        word = pattern.sub('', word)
        if len(word) != 0:
            wordsListAll.append(word)
CountDict = dict(Counter(wordsListAll))
keyCount = list(CountDict.keys())
valueCount = list(CountDict.values())

A = Bayesianclassifier() # 实例化
A.Fitting_model(train_set, train_table, 1)#     # 拟合模型

predicted_table = A.predicted(test_set, 1)  #预测数据分类、  返回预测标签
table_0 = []
table_1 = []
table_n1 = [] #-1标签
for table in predicted_table:
    if table == 0:
        table_0.append(table)
    elif table == 1:
        table_1.append(table)
    else:
        table_n1.append(table)
strprecision, strrecall = A.evaluation_model(predicted_table)  # 评估模型 ;召回、准确
positive, neutral, negative, expect = A.statistical_data()  # 统计数据 正、中、负、期望
positive = int(float(str(positive)[:6])*1000)+1
neutral = int(float(str(neutral)[:6])*1000)+1
negative = int(float(str(negative)[:6])*1000)+1

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getData1')
def getData1():
    key = ['positive', 'neutral', 'negative']
    value = [len(table_0),len(table_1),len(table_n1)]
    return json.dumps({"key":key,"value":value},ensure_ascii=False)

@app.route('/getData2')
def getData2():
    key = ['positive','neutral','negative']
    value = [positive, neutral, negative]
    return json.dumps({"key":key,"value":value},ensure_ascii=False)

@app.route('/getData3')
def getData3():
    global keyCount,valueCount
    return json.dumps({"key":keyCount[:600],"value":valueCount[:600]},ensure_ascii=False)

@app.route('/getData4')
def getData4():
    global keyCount,valueCount
    return json.dumps({"key":keyCount,"value":valueCount},ensure_ascii=False)

app.run(host='127.0.0.1',port=8090,debug=1)


