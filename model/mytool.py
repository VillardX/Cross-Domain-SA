import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')


def k_s_fold(data, k, y_name='label'):
    '''
        对原始数据进行分层kfold，即保证每个子集Y取各值的比例和原始数据集相接近
        输入：
            data:原始数据，为dataframe
            k：要将原始数据平分的子集个数
            y_name：标签列列名,为字符串
        输出：
            train_index_list:[train1_id_list,...,traink_id_list]
            test_index_list:[test1_id_list,...,testk_id_list]
    '''
    # 定位X与y
    y = data[y_name]
    X = data[[col for col in data.columns if col != y_name]]

    SKF = StratifiedKFold(n_splits=k,random_state=None, shuffle=False)  # , 
    SKF_spli = SKF.split(X, y)
    train_index_list = []
    test_index_list = []
    for train_index, test_index in SKF_spli:
        train_index_list.append(train_index)
        test_index_list.append(test_index)
    return train_index_list, test_index_list


def idlize(data, seg_name='seg_result'):
    '''
        将分词结果id化
        输入
            seg_name：需要将所分词列表化的列列名，该列下的每一元素形同'只能 说 游戏 做 的 确实 不怎么 如 人意 ， 慢慢 调整 吧',以空格分开
        输出
            data:添加了word_num列的更新data
            dic_id2word:序号到词的映射词典
            dic_word2id:词到序号的映射词典
    '''

    # 构建词典对应表
    seg = data[seg_name]
    seg_spli = seg.str.split(' ')
    seg_list = seg_spli.to_list()

    flatten_seg_list = [each_word for each_sentence_list in seg_list for each_word in each_sentence_list]  # 展开成一个list
    word_list = set(flatten_seg_list)  # 去重
    word_list = list(word_list)#set是无序的,每次运行都会有不一样的顺序
    word_list.sort()#用list的sort功能排序,这样每次就能得到相同的结果了

    # 构建词典，并将出现的词填入词典
    dic_id2word = {i + 1: j for i, j in enumerate(word_list)}

    dic_word2id = dict(zip(dic_id2word.values(), dic_id2word.keys()))  # 数为值单词为键

    # 构建每条评论的分词id映射
    seg_id_list = []
    for each_seg in seg_list:
        temp = []
        for each_word in each_seg:
            temp.append(dic_word2id[each_word])
        seg_id_list.append(temp)

    # 将各条评论id映射加入元数据
    data['seg_id_list'] = seg_id_list
    data['seg_word_list'] = seg_spli

    data['word_num'] = data['seg_id_list'].apply(lambda x: len(x))  # 每条评论的词的个数，对应content_len字数

    return data, dic_id2word, dic_word2id


class tf_idf_method:
    def __init__(self, data, stp_list, seg_col_name='seg_result',
                 other_x_name=['fun', 'up', 'down', 'play_time', 'phone_code', 'content_len']):
        '''
            初始化
                data:原始数据，为dataframe
                seg_col_name:分词结果所在列列名，为str
                stp_list:停止词词典，为list
                other_x_name：非文字特征，为list
                feature_name:特征名称，初始化为非文字特征other_x_name，后面通过tf_idf_vec操作会加上文字特征，为list
        '''
        self.data = data
        self.seg_col_name = seg_col_name
        self.stop_word = stp_list
        self.other_x_name = other_x_name  # 非文字特征
        self.feature_name = other_x_name
        self.final_X = None  # 最终处理好的X数据


    def tf_idf_vec(self, max_f=5000):
        """
            对分词结果进行tf-idf化
            输入：
                max_f = 最大维数，默认5000维
            输出：
                self.final_data：最终处理好的数据，其他特征在前、文字特征在后的sparse matrix

        """
        corpus = self.data[self.seg_col_name]  # 需要tf-idf向量化的列

        vectorizer = TfidfVectorizer(stop_words=self.stop_word, max_features=max_f)  # 初始化模型,max_features先定5000维，后续可调
        seg_X = vectorizer.fit_transform(corpus)  # 得到的tf-idf向量，是个sparse matrix

        self.feature_name = self.feature_name + vectorizer.get_feature_names()  # 将文字特征加入特征名

        other_x_value = self.data.loc[:, self.other_x_name].values  # 转为np.array
        other_x_sparse = sp.csr_matrix(other_x_value)  # 离散化

        # 非文本特征与文本特征合并，注意先后顺序
        total_X = sp.hstack([other_x_sparse, seg_X]).tocsr()  # 转为csr矩阵，后期交叉验证，多会做行切片

        self.final_X = total_X


class lgbmodel():
    def __init__(self, X, y, X_name_list):
        '''
            初始化
                X：为特征，sparse_matrix
                y：标签，pd.series
                X_name_list:各特征名，为list
        '''
        # 特征名字典
        self.feature_dict = {X_name_list[i]: ('x' + str(i + 1)) for i in
                             range(len(X_name_list))}  # lgb不支持中文特征，所以以xi作为代号，形如{'x1':'fun'...}
        # 数据初始化
        self.train = lgb.Dataset(X, label=y, feature_name=self.feature_dict.values(),
                                 categorical_feature=[self.feature_dict['phone_code']])  # 手机型号是分类属性
        #         self.train_X = lgb.Dataset(X)
        #         self.train_y = lgb.Dataset(y)
        #         self.train = lgb.Dataset(self.train_X, label=self.train_y, feature_name=self.feature_dict.values(), categorical_feature=[self.feature_dict['phone_code']])

        # 模型初始化
        self.params = {'max_depth': 10, 'min_data_in_leaf': 5,
                       'learning_rate': 0.1, 'num_leaves': 1024, 'metric': ['binary_logloss', 'binary_error'],
                       'objective': 'binary', 'nthread': 4, 'verbose': -1, 'feature_fraction': 0.8,
                       'feature_fraction_seed': 1}  # 参数'num_leaves': 35, 'lambda_l1': 0.1, 'lambda_l2': 0.2,
        self.lgb_model = None  # 模型

    def fit(self, num_boost=1000):
        '''
            训练
        '''
        self.lgb_model = lgb.train(self.params, self.train, num_boost, verbose_eval=100)  # verbose_eval迭代多少次打印
