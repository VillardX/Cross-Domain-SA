import os
import random
from statistics import mode
import sys

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction,AdamW, get_linear_schedule_with_warmup
from transformers import BertModel

#自用包
from data_loader import GetLoader
from model import BERT_DANN_model
from test import test

model_root = '../temp_model_output'#每次训练完一个epoch储存当前model的路径
PLM_root = '../PLMs'#预训练模型位置
# cuda = True
# cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('当前device')
print(device)

lr = 1e-3
BATCH_SIZE_source = 16#游戏数据量较大，设置较大batch
BATCH_SIZE_target = 4096#餐饮数据量较小，设置较小batch
n_epoch = 10
MAX_LEN = 512#BERTmodel截断长度
NUM_WORKERS = 0#Dataloader参数

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

def gen_PLM_path(PLM_name,PLM_root):
    '''
        根据PLM名称生成相应路径
    '''
    PLM_name = PLM_name.replace('-','_')#默认模型名称为xx-xx-xx而路径名为xx_xx_xx
    PLM_path = PLM_root + '/' + PLM_name + '/'
    return PLM_path

#PLM模型设置
model_name = 'bert-base-chinese'
MODEL_PATH = gen_PLM_path(model_name,PLM_root)
print(MODEL_PATH)
# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# b. 导入配置文件
model_config = BertConfig.from_pretrained(MODEL_PATH)
# 修改配置 - 最后一层的输出是否需要下面两项
model_config.output_hidden_states = False
model_config.output_attentions = False
# 通过配置和路径导入模型
#bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)

# load data
#后续改成多模块
dt_game = pd.read_csv('../data/game2polarity.csv',encoding='utf-8')#手游数据
dt_game['review'] = dt_game['review'].apply(lambda x : str(x))
dt_res = pd.read_csv('../data/rest2polarity.csv',encoding='utf-8')#餐厅数据
print('数据载入完成')


print('将原始数据载入Dataloader')
#GetLoader()得到的结果其实为一个nn.Dataset，后续还要放进nn.DataLoader得到一个DataLoader实例进行batch_BP
#此处以手游数据为源，以餐厅数据为目标
dataset_source = GetLoader(
    dt=dt_game,
    tokenizer=tokenizer,
    max_len=MAX_LEN)
#根据Dataset创建DataLoader实例
dataloader_source = DataLoader(
    dataset=dataset_source,
    batch_size=BATCH_SIZE_source,
    shuffle=True,
    num_workers=NUM_WORKERS
)

dataset_target = GetLoader(
    dt=dt_res,
    tokenizer=tokenizer,
    max_len=MAX_LEN)
dataloader_target = DataLoader(
    dataset=dataset_target,
    batch_size=BATCH_SIZE_target,
    shuffle=True,
    num_workers=NUM_WORKERS
)

# load model

my_net = BERT_DANN_model(
    num_inputs=768,
    num_hiddens=768,
    MODEL_PATH=MODEL_PATH,
    model_config=model_config
    )

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()#损失函数初始化
loss_domain = torch.nn.CrossEntropyLoss()#损失函数初始化
#参数reduction默认为"mean"，表示对所有样本的loss取均值，最终返回只有一个值
#参数reduction取"none"，表示保留每一个样本的loss
#参考https://blog.csdn.net/u012633319/article/details/111093144

my_net = my_net.to(device)
loss_class = loss_class.to(device)
loss_domain = loss_domain.to(device)

for p in my_net.parameters():
    p.requires_grad = True

# training
print('开始训练')
best_accu_t = 0.0
for epoch in range(n_epoch):####应该设置train和eval模式

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    #每一轮我都把原始的dataloader变成一个迭代器
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)
    #创造了一个源数据的迭代器，因为dataloader_source是nn.Dataset实例具有迭代属性，所以可以据此建立一个迭代器，可以使用next()更新新一份数据
    #当然也可以使用enumerate()访问dataset，详情参考https://blog.csdn.net/weixin_44533869/article/details/110856518

    for i in range(len_dataloader):
        
        #各一个batch训练都要改变GRL超参
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1#计算GRL的参数

        # training model using source data
        data_source = data_source_iter.next()#得到相关数据
        #各部分数据结构
        # {
        #     'sentiment_label':torch.tensor(sentiment_label,dtype=torch.long),
        #     'input_ids': encoding['input_ids'].flatten(),
        #     'attention_mask': encoding['attention_mask'].flatten(),
        #     'token_type_ids': encoding['token_type_ids'].flatten(),
        # }
        s_sentiment_label = data_source['sentiment_label'].to(device)
        s_input_ids = data_source['input_ids'].to(device)
        s_attention_mask = data_source['attention_mask'].to(device)
        s_token_type_ids = data_source['token_type_ids'].to(device)

        my_net.zero_grad()

        domain_label = torch.zeros(BATCH_SIZE_source).long().to(device)#domain的label

        class_output, domain_output = my_net(s_input_ids,s_attention_mask,s_token_type_ids,alpha)
        err_s_label = loss_class(class_output, s_sentiment_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()

        t_sentiment_label = data_target['sentiment_label'].to(device)
        t_input_ids = data_target['input_ids'].to(device)
        t_attention_mask = data_target['attention_mask'].to(device)
        t_token_type_ids = data_target['token_type_ids'].to(device)

        domain_label = torch.ones(BATCH_SIZE_target).long().to(device)#domain的label

        _, domain_output = my_net(t_input_ids,t_attention_mask,t_token_type_ids,alpha)
        err_t_domain = loss_domain(domain_output, domain_label)

        #三部分的error加总得到总的err
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        torch.save(my_net, '{0}/game2meituan_model_epoch_current.pth'.format(model_root))

    #每一个epoch解释进行测试
    print('\n')
    accu_s = test('game',tokenizer,max_len=MAX_LEN)
    print('Accuracy of the %s dataset: %f' % ('game', accu_s))
    accu_t = test('meituan',tokenizer,max_len=MAX_LEN)
    print('Accuracy of the %s dataset: %f\n' % ('meituan', accu_t))
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        torch.save(my_net, '{0}/game2meituan_model_epoch_best.pth'.format(model_root))
