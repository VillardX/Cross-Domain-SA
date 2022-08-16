from ast import arg
import json
import random
import copy
import logging
import argparse
import os
from tokenize import Token
from numpy import source
import yaml
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange
from utils import set_seeds
from post_training_model import PostTrainingModel
import sys

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import WEIGHTS_NAME, CONFIG_NAME

TIME_STAMP = datetime.now().strftime('%Y-%m-%dT%H:%M')
if not os.path.exists(f'./checkpoints/post_{TIME_STAMP}'):
    print('not exist')
    os.mkdir(f'./checkpoints/post_{TIME_STAMP}')
    
config = yaml.load(open('./post_logger.yml'), Loader=yaml.FullLoader)
config['handlers']['file_info']['filename'] = f'./checkpoints/post_{TIME_STAMP}/train.log'
logging.config.dictConfig(config)
logger = logging.getLogger()

class PostTraining(Dataset):
    def __init__(self, MAX_LEN, tokenizer):
        self.tokenizer = tokenizer
        self.sample_counter = 0#采样数量
        self.max_seq_length = MAX_LEN

        source_data = pd.read_csv('../data/unlabel_game.csv',encoding='utf-8',engine='python')
        source_data = source_data[source_data['review'].str.len()>100]['review'].tolist()#长度多余100的才放进采样
        target_data = pd.read_csv('../data/unlabel_rest.csv',encoding='utf-8',engine='python')
        target_data = target_data['review'].tolist()

        self.mix_data = source_data
        

        _target_data = []
        target_total = 0
        while target_total < 100000:#后续可以设置成和source数据量一样
            _target_data.extend(random.sample(target_data,20000))
            target_total = len(_target_data)
        random.shuffle(_target_data)
        self.target_data = _target_data
    
    def __len__(self):
        return len(self.target_data)
        #本质上就是生成的10w条target_data的索引，且默认target的数据量小于source，所以有一部分source是取不到的
    
    def __getitem__(self, idx):
        guid = self.sample_counter
        self.sample_counter += 1

        #分别为原始的句1、原始的句2，即没有经过tokenize的原始文本
        # is_mix:两句句子是否来自不同领域,取1表示两句句子来自不同域；取0表示两句句子都来自目标域
        # 句1的领域、句2的领域：取'target'和'mix'
        t1, t2, is_mix, t1_domain_label, t2_domain_label= self.random_sent(idx)

        #做一些文本清洗后进行分词
        #tokenize后原始文本变为一个list
        #e.g."Hello, my dog is cute"→['hello', ',', 'my', 'dog', 'is', 'cut', '##e']
        tokens_a = self.tokenizer.tokenize(t1)#clean(t1)#只是作分词输入"Hello, my dog is cute"→['hello', ',', 'my', 'dog', 'is', 'cut', '##e']
        tokens_b = self.tokenizer.tokenize(t2)#clean(t2)

        #构建了一个类实例，此实例中只有mlm_labels没有定义，后面会继续处理
        example = InputExample(guid=guid, tokens_a=tokens_a, tokens_b=tokens_b, is_mix=is_mix,
                                t1_domain_label=t1_domain_label, t2_domain_label=t2_domain_label)
        
        features = convert_example_to_features(example, self.tokenizer, self.max_seq_length)
        
        #返回的内容还要改改#####
        tensors = (torch.tensor(features.raw_input_ids),
                    torch.tensor(features.masked_input_ids),
                    torch.tensor(features.input_mask),
                    torch.tensor(features.segment_ids),
                    torch.tensor(features.is_mix),
                    torch.tensor(features.mlm_pred_positions),
                    torch.tensor(features.mlm_true_ids),
                    torch.tensor(features.mlm_weights))

        return tensors
    
    def random_sent(self, idx):
        '''
            tx_domain_label:取'target'或'mix'，target代表取自目标域，而mix代表取自源域？？？
                此标签用于后续判断是否要做mlm
        '''
        t1, t2 = self.get_target_line(idx)
        t1_domain_label = t2_domain_label = 'target'

        if random.random() > 0.5:
            is_mix = 0#说明是非混合的数据
        else:
            #再取一个随机数，如果大于0.5则把第一句换成source域的句子；否则把第二局换成source的句子
            if random.random() > 0.5:
                t1 = self.get_other_line(idx)
                t1_domain_label = 'mix'#亦即说明句1来自源域
            else:
                t2 = self.get_other_line(idx)
                t2_domain_label = 'mix'#亦即说明句2来自源域
            is_mix = 1#说明是混合数据
        
        assert len(t1) > 0#如果len(t1)>0则继续运行，否则就报错
        assert len(t2) > 0
        return t1, t2, is_mix, t1_domain_label, t2_domain_label

    def get_target_line(self, idx):
        '''
            从self.target_data中随机取两条文本，即从target领域随机抽取两条文本
        '''
        t1 = self.target_data[idx]

        if idx == len(self.target_data) - 1:#第一句为列表最后一句，则第二句回到开头
            t2 = self.target_data[0]
        else:
            t2 = self.target_data[idx + 1]
        
        return t1, t2

    def get_other_line(self, idx):
        '''
            从self.mix_data亦即源域中取出一条数据
        '''
        line = self.mix_data[idx]
        
        assert len(line) > 0
        return line

class InputExample:
    def __init__(self, 
                guid, 
                tokens_a, 
                tokens_b=None, 
                is_mix=None, 
                mlm_labels=None, 
                t1_domain_label=None, 
                t2_domain_label=None):
        
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_mix = is_mix # A, B sentence是否来自同一领域
        self.mlm_labels = mlm_labels  # masked words for language model
        self.t1_domain_label = t1_domain_label
        self.t2_domain_label = t2_domain_label

class InputFeatures:
    def __init__(self, 
                raw_input_ids,
                masked_input_ids, 
                input_mask, 
                segment_ids, 
                is_mix, 
                mlm_pred_positions,
                mlm_true_ids,
                mlm_weights):
        '''
            mlm_input_ids:mlm以后的token_id列表
            input_mask:别名：attention_mask,识别每个token是否为padding，非padding为1，否则为0
            segment_ids:别名token_type_ids,判断是第一句0还是第二句1，padding也记为0
            is_mix：判断是否是混合域还是都来自目标域
        '''
        self.raw_input_ids = raw_input_ids
        self.masked_input_ids = masked_input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_mix = is_mix
        self.mlm_pred_positions = mlm_pred_positions
        self.mlm_true_ids = mlm_true_ids
        self.mlm_weights = mlm_weights

def random_word2(tokensAB, whether_do_mlm_listAB,tokenizer):
    '''
        tokensAB和whether_do_mlm_listAB等长--
        对concate到一起的两句句子做mlm，仅针对target-domain做

        return 
            masked_tokensAB：完成mlm词被[MASK]覆盖以后的tokens-list
            pred_positions:被mask的token的位置list，padding至同masked_tokensAB等长
            true_tokens：被mask的token的ground-truth-token的list同pred_positions等长，padding至同masked_tokensAB等长
            #mlm_weights：权重，真正被mask的词权重为1，而被pad的为0一个列表，与上述两个列表对应，原始位置全是被mask的索引位置为1，被padding的位置为0
    '''
    pred_positions,true_tokens = [],[]
    masked_tokensAB = [t for t in tokensAB]#浅拷贝一份
    for i,token in enumerate(masked_tokensAB):
        if whether_do_mlm_listAB[i] == 1:#说明是target-domain
            prob = random.random()
            # mask token with 15% probability
            # 此处使用了15%概率而不是比例，和原文bert有一定差别#后面计算mlm损失时，如果采用概率处理，则padding得maxlen才最保险，否则担心会溢出
            if prob < 0.15:#说明要进行mask
                prob /= 0.15
                #此处沿用bert原文的80，10，10配比
                if prob < 0.8:masked_tokensAB[i] = '[MASK]'
                elif prob < 0.9:#随机抽一个字
                    masked_tokensAB[i] = random.choice(list(tokenizer.vocab.items()))[0]#[0]是具体词，[1]是该词的id
                #还有10%不作处理，保持原样
            pred_positions.append(i)
            true_tokens.append(token)
    # Predictions of padded tokens will be filtered out in the loss via
    # multiplication of 0 weights
    # padding词元的预测将通过乘以0权重在损失中过滤掉,总长度即masked_tokensAB的长度
    # mlm_weights = ([1.0]*len(pred_positions) + [0.0]*(len(tokensAB)-len(pred_positions)))#
    return masked_tokensAB,pred_positions,true_tokens

def _concate_tokensAB(tokens_a,tokens_b,a_domain_label,b_domain_label):
    '''
        tokens_a：经过tokenizer.tokenize获得的token_list
        tokens_b
        a_domain_label：tokens_a所在的领域，取'mix'和'target'二值
        b_domain_label：

        return:
            concate_tokensAB_with_special_token：经过【CLS】和【SEP】连接的tokenlist
            whether_do_mlm_listAB：和concate_tokensAB_with_special_token等长，取0：说明来自源于，不做mlm；取1说明来自目标域，可能会做mlm  
            segment_ids：返回concate_tokensAB_with_special_token的segment_ids        
    '''
    concate_tokensAB_with_special_token = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
    d = {'mix':0,'target':1}

    segment_ids = [0]+[0]*len(tokens_a)+[0]+[1]*len(tokens_b)+[1]
    whether_do_mlm_listAB = [0] + [d[a_domain_label]]*len(tokens_a) + [0] + [d[b_domain_label]]*len(tokens_b) + [0]
    return concate_tokensAB_with_special_token,whether_do_mlm_listAB,segment_ids

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
        Truncates a sequence pair in place to the maximum length.
        此处是直接应用tokens_a和tokens_b两个列表，所以不会返回，直接在原始变量上做了修改
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()#移除最后一个元素
        else:
            tokens_b.pop()

def convert_example_to_features(example, tokenizer, max_seq_length):
    '''
        example:一个InputExample实例
    '''
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    
    #裁剪到最大长度，但是未到最大长度的没有做补齐
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)#-3是为了留出[CLS]和两个[SEP]

    #将两句句子拼接起来
    #这里放的是经过MASK后的token-a和token-b
    tokens,whether_do_mlm_listAB,segment_ids = _concate_tokensAB(tokens_a,tokens_b,example.t1_domain_label,example.t2_domain_label)
    #相关input的定义可以参考https://yiyele.blog.csdn.net/article/details/89882529?spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-10-89882529-blog-122306758.t5_layer_targeting_s&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-10-89882529-blog-122306758.t5_layer_targeting_s&utm_relevant_index=17
    raw_input_ids = tokenizer.convert_tokens_to_ids(tokens)#经token转为id
    input_mask = [1] * len(raw_input_ids)#1表示这些词都不是通过padding得到的

    #获得mlm后的token-list，
    # 被mlm的位置索引，
    # 被mlm的实际groud-truth-token，
    # 表示的mlm位置的权重未mlm位置的权重未0计算loss的时候不考虑
    masked_tokens,mlm_pred_positions,mlm_true_tokens = random_word2(tokens, whether_do_mlm_listAB,tokenizer)
    
    #masked_input_ids同raw_input_ids等长
    #而mlm_pred_positions,mlm_true_ids,mlm_weights等长
    #padding的时候注意分开处理
    masked_input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    mlm_true_ids = tokenizer.convert_tokens_to_ids(mlm_true_tokens)
    mlm_weights = [1.0]*len(mlm_pred_positions)#至mlm的位置权重为1后面padding设置其他位置为0  
    
    #padding
    while len(raw_input_ids) < max_seq_length:
        raw_input_ids.append(0)
        masked_input_ids.append(0)
        input_mask.append(0)#0表示这些token都是通过padding得到的，属于tokenizer内置项
        segment_ids.append(0)#识别为哪一句，第一句为0，第二句为1，padding默认为0

    assert len(raw_input_ids) == max_seq_length
    assert len(masked_input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    #padding2
    while len(mlm_pred_positions) < max_seq_length:
        mlm_pred_positions.append(0)
        mlm_true_ids.append(0)
        mlm_weights.append(0)

    if example.guid < 5:#只取前1条数据作展示
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("masked_tokens: %s" % " ".join(
                [str(x) for x in masked_tokens]))

        logger.info("raw_input_ids: %s" % " ".join([str(x) for x in raw_input_ids]))
        logger.info("masked_input_ids: %s" % " ".join([str(x) for x in masked_input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("mlm_pred_positions: %s " % (mlm_pred_positions))
        logger.info("mlm_true_ids: %s " % (mlm_true_ids))
        logger.info("mlm_weights: %s " % (mlm_weights))
        logger.info("Is mix domain label: %s " % (example.is_mix))
        logger.info("t1 domain label: %s " % (example.t1_domain_label))
        logger.info("t2 domain label: %s " % (example.t2_domain_label))

    #构造一个数据类，输入即为这个类的每一个属性
    #输出结果已经添加【CLS】、【SEP】；被mask过；并完成padding
    features = InputFeatures(raw_input_ids=raw_input_ids,
                            masked_input_ids=masked_input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             is_mix=example.is_mix,
                             mlm_pred_positions=mlm_pred_positions,
                             mlm_true_ids=mlm_true_ids,
                             mlm_weights=mlm_weights)
    return features
