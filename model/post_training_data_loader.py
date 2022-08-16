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

# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.WARNING)

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

        # with open(f'../data/sports.json', 'r', encoding='utf-8') as f:
        #     sports_data = json.load(f)
        # with open(f"../data/tv.json", 'r', encoding='utf-8') as f:
        #     tv_data = json.load(f)
        # with open(f"../data/news_content_number.txt", 'r', encoding='utf-8') as f:
        #     mix_data = f.readlines()

        # sports_data = [line['text'] for line in sports_data if len(line['text']) > 5]
        # tv_data = [line['text'] for line in tv_data if len(line['text']) > 5]
        # self.mix_data = [line for line in mix_data if len(line) > 10]

        # _sports_data, _tv_data = [], []
        # sports_total, tv_total = 0, 0
        # while sports_total != 8000 or tv_total != 8000:   
        #     _sports_data.extend(random.sample(sports_data, 8000-sports_total))
        #     _tv_data.extend(random.sample(tv_data, 8000-tv_total))

        #     _sports_data = [line for line in _sports_data if len(self.tokenizer.tokenize(line)) > 2]
        #     _tv_data = [line for line in _tv_data if len(self.tokenizer.tokenize(line)) > 2]

        #     sports_total = len(_sports_data)
        #     tv_total = len(_tv_data)

        # total = _sports_data + _tv_data
        # print(len(total))
        # self.target_data = []
        # #total是从两个领域分别有放回地抽样了8000条并整合在了一起
        # #self.target_data基于total洗牌洗了10次，最后把10个list并在了一起
        # for _ in range(10):
        #     temp = copy.deepcopy(total)
        #     random.shuffle(temp)
        #     self.target_data.extend(temp)
    
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


def random_word(tokens, tokenizer, domain_label):
    '''
        对目标域做mlm
        tokens：tokenizer.tokenize()所得list e.g.'hello', ',', 'my', 'dog', 'is', 'cut', '##e']
        domain_label：标识改句子是目标域还是源域，只对目标域句子做mlm
        tokenizer:PLM的tokenize工具函数
        输出：通过mlm更新后的tokens-list
            对应每个tokens-list的output-label，只记录需要预测的token的ground truth idx，其他token记录为-100对应[UNK]
            e.g. 原始['hello', ',', 'my', 'dog', 'is', 'cut', '##e']
                mlm后['hello', ',', '[MASK]', 'cat', 'is', 'cut', '##e']
                output-label:[-100,-100,103对应[MASK],13030对应dog,-100,-100,8154对应##e]
    '''
    output_label = []
    
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        # 此处使用了15%概率而不是比例，和原文bert有一定差别
        if domain_label == 'target':
            if prob < 0.15:#说明要进行mask
                prob /= 0.15 

                #此处沿用bert原文的80，10，10配比
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                elif prob < 0.9:#随机抽一个字
                    tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]#[0]是具体词，[1]是该词的id
                #还有10%不作处理，保持原样

                # append current token to output (we will predict these later)
                try:
                    output_label.append(tokenizer.vocab[token])#返回mlm需要预测的ground truth token的idx
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # no masking token (will be ignored by loss function later)
                # 剩下的不做mask，赋值为-100，对应tokenizer词库中的[UNK]
                output_label.append(-100)
        else:#源域中的数据也不做mask预测，故赋值-100
            output_label.append(-100)

    return tokens, output_label

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
    

    # ##############
    # #mlm后的词序列list；以及对应的需要预测的词的真正的id，剩下的id设为-100，即对应[UNK]
    # tokens_a, t1_label = random_word(tokens_a, tokenizer, example.t1_domain_label)
    # tokens_b, t2_label = random_word(tokens_b, tokenizer, example.t2_domain_label)



    #t1_label和t2_label之间再加上[SEP]和[CLS]三个符位
    # mlm_label_ids = ([-100] + t1_label + [-100] + t2_label + [-100])#两个[SEP]和[CLS]也不需要预测

    #初始化
    #这里放的是经过MASK后的token-a和token-b
    # tokens = []
    # segment_ids = []
    # tokens.append("[CLS]")
    # segment_ids.append(0)
    #处理第一句 #还可以优化，直接进行加法
    # for token in tokens_a:
        # tokens.append(token)
        # segment_ids.append(0)
    # tokens.append("[SEP]")
    # segment_ids.append(0)

    # try:
        # assert len(tokens_b) > 0
    # except:
        # print(example.tokens_a)
        # print(example.tokens_b)
        # print(example.is_mix)
    #处理第二句 #还可以优化，直接进行加法
    # for token in tokens_b:
        # tokens.append(token)
        # segment_ids.append(1)
    # tokens.append("[SEP]")
    # segment_ids.append(1)

    
    
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

def main(args):
    set_seeds()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PostTrainingModel(args, device).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_dataset = PostTraining(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size,num_workers=8)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler是对optimizer学习率的适时调整
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", t_total)
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Learning rate = %d", args.learning_rate)
    
    global_step = 0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    model.train()#开启训练模式
    for epoch in train_iterator:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()#记得和model.zero_grad()进行区别

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, mlm_label_ids, is_mix = batch
            
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=mlm_label_ids, next_sentence_label=is_mix)
            loss = outputs[0]
        
            loss.backward()#保留每个节点的梯度，每次累加
            tr_loss += loss.item()#显示更高精度的loss
            nb_tr_steps += 1
            
            #到需要更新的时候将之前的梯度积攒在一起一并更新
            #参https://zhuanlan.zhihu.com/p/445009191
            #累积一段时间，再一并跟新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()#先调整梯度，进行回传
                scheduler.step() #而后调整optimizer的学习率
                model.zero_grad()#最后模型梯度重置为0，进行新的计算
                global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
                
            # sys.stdout.write('\r epoch: %d, [iter: %d / all %d]' \
            #   % (epoch, step + 1, len(epoch_iterator)))
            # sys.stdout.flush()

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        # Save a trained model

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    print(WEIGHTS_NAME)
    print(CONFIG_NAME)
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    # model_to_save.args.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)
    logger.info(f'train loss = {tr_loss / global_step}')
    logger.info(f'global steps = {global_step}')
    logger.info("=========== Saving fine - tuned model ===========")



# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
    

#     ## Required parameters
#     parser.add_argument("--train_corpus",
#                         default='../data/',
#                         type=str,
#                         help="The input train corpus. sports or tv")
#     parser.add_argument("--mix_domain",
#                         default='../data/',
#                         type=str,
#                         help="The input train corpus. sports or tv")
#     parser.add_argument("--bert_model", default='../PLMs/bert_base_chinese', type=str,
#                         help="Bert pre-trained model selected in the list: bert-base-uncased, "
#                              "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
#     # parser.add_argument("--output_dir",
#     #                     default=f'./checkpoints/post_{TIME_STAMP}/',
#     #                     type=str,
#     #                     help="The output directory where the model checkpoints will be written.")

#     ## Other parameters
#     parser.add_argument("--max_seq_length",
#                         default=512,
#                         type=int,
#                         help="The maximum total input sequence length after WordPiece tokenization. \n"
#                              "Sequences longer than this will be truncated, and sequences shorter \n"
#                              "than this will be padded.")
#     parser.add_argument("--do_train",
#                         action='store_true',
#                         help="Whether to run training.")
#     parser.add_argument("--train_batch_size",
#                         default=16,
#                         type=int,
#                         help="Total batch size for training.")
#     parser.add_argument("--learning_rate",
#                         default=2e-5,
#                         type=float,
#                         help="The initial learning rate for Adam.")
#     parser.add_argument("--num_train_epochs",
#                         default=5.0,
#                         type=float,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--warmup_proportion",
#                         default=0.1,
#                         type=float,
#                         help="Proportion of training to perform linear learning rate warmup for. "
#                              "E.g., 0.1 = 10%% of training.")
#     parser.add_argument("--no_cuda",
#                         action='store_true',#设置了一个开关https://blog.csdn.net/qq_31347869/article/details/104837836
#                         default=False,
#                         help="Whether not to use CUDA when available")
#     parser.add_argument("--local_rank",
#                         type=int,
#                         default=-1,
#                         help="local_rank for distributed training on gpus")
#     parser.add_argument('--seed',
#                         type=int,
#                         default=42,
#                         help="random seed for initialization")
#     parser.add_argument('--gradient_accumulation_steps',
#                         type=int,
#                         default=1,
#                         help="Number of updates steps to accumualte before performing a backward/update pass.")
#     parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
#     parser.add_argument("--max_steps", default=10000, type=int, help="Maximum steps size")
#     parser.add_argument("--hidden_size", default=768, type=int, help="Hidden Vector size")

#     args = parser.parse_args()

#     main(args)




# parser = argparse.ArgumentParser()
    

# ## Required parameters
# parser.add_argument("--train_corpus",
#                     default='../data/',
#                     type=str,
#                     help="The input train corpus. sports or tv")
# parser.add_argument("--mix_domain",
#                     default='../data/',
#                     type=str,
#                     help="The input train corpus. sports or tv")
# parser.add_argument("--bert_model", default='../PLMs/bert_base_chinese', type=str,
#                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
#                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
# parser.add_argument("--output_dir",
#                     default=f'./checkpoints/post_{TIME_STAMP}/',
#                     type=str,
#                     help="The output directory where the model checkpoints will be written.")

# ## Other parameters
# parser.add_argument("--max_seq_length",
#                     default=512,
#                     type=int,
#                     help="The maximum total input sequence length after WordPiece tokenization. \n"
#                             "Sequences longer than this will be truncated, and sequences shorter \n"
#                             "than this will be padded.")
# parser.add_argument("--do_train",
#                     action='store_true',
#                     help="Whether to run training.")
# parser.add_argument("--train_batch_size",
#                     default=8,
#                     type=int,
#                     help="Total batch size for training.")
# parser.add_argument("--learning_rate",
#                     default=2e-5,
#                     type=float,
#                     help="The initial learning rate for Adam.")
# parser.add_argument("--num_train_epochs",
#                     default=5.0,
#                     type=float,
#                     help="Total number of training epochs to perform.")

# parser.add_argument("--num_steps",
#                     default=100000,
#                     type=int,
#                     help="Total number of training batch times to perform.")
# parser.add_argument("--warmup_proportion",
#                     default=0.1,
#                     type=float,
#                     help="Proportion of training to perform linear learning rate warmup for. "
#                             "E.g., 0.1 = 10%% of training.")
# parser.add_argument("--no_cuda",
#                     action='store_true',#设置了一个开关https://blog.csdn.net/qq_31347869/article/details/104837836
#                     default=False,
#                     help="Whether not to use CUDA when available")
# parser.add_argument("--local_rank",
#                     type=int,
#                     default=-1,
#                     help="local_rank for distributed training on gpus")
# parser.add_argument('--seed',
#                     type=int,
#                     default=42,
#                     help="random seed for initialization")
# parser.add_argument('--gradient_accumulation_steps',
#                     type=int,
#                     default=1,
#                     help="Number of updates steps to accumualte before performing a backward/update pass.")
# parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
# parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
# parser.add_argument("--max_steps", default=10000, type=int, help="Maximum steps size")
# parser.add_argument("--hidden_size", default=768, type=int, help="Hidden Vector size")

# args = parser.parse_args(args=[])#notbook中要加args=[]

# main(args)