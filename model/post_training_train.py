from ast import arg
import json
import random
import copy
import logging
import argparse
import os
from sched import scheduler
from tokenize import Token
from numpy import source
import yaml
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange
from utils import set_seeds
from post_training_model import PostTrainingModel
from post_training_data_loader import PostTraining
import sys

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import WEIGHTS_NAME, CONFIG_NAME

TIME_STAMP = datetime.now().strftime('%Y-%m-%dT%H:%M')  
if not os.path.exists(f'./checkpoints/post_{TIME_STAMP}'):
    # logger.info(f'./checkpoints/post_{TIME_STAMP} not existed, so create it')
    os.mkdir(f'./checkpoints/post_{TIME_STAMP}')
else:print('existed-train-part')


config = yaml.load(open('./post_logger.yml'), Loader=yaml.FullLoader)
config['handlers']['file_info']['filename'] = f'./checkpoints/post_{TIME_STAMP}/train.log'

logging.config.dictConfig(config)
logger = logging.getLogger()

def post_train(train_loader, net, device, num_steps,optimizer,schedule):
    '''
        正式开始训练
        输入：
            train_loader：对应data_loader
            net：对应网络
            device：设备情况
            num_steps：代替epoch，直接迭代次数
            optimizer：优化器
            schedule:warm_up优化器
    '''
    # net = net.to(device)
    # loss = loss.to(device)
    # trainer = torch.optim.Adam(net.parameters(), lr=2e-5)#可以设置warm_up但是没必要，这里是further_pretrain
    # step, timer = 0, d2l.Timer()
    #animator = d2l.Animator(xlabel='step', ylabel='loss',xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语⾔模型损失的和，下⼀句预测任务损失的和，句⼦对的数量，计数
    #metric = d2l.Accumulator(4)
    num_steps_reached = False
    tr_loss = 0#总的loss，不用优化，最后仅用于展示
    # step = 0
    iter_train = iter(train_loader)
    for step in trange(num_steps):
    # while step < num_steps and not num_steps_reached:
        try:
            batch_dt = iter_train.next()
        except:
            print('当前data_loader已用尽或者本次为第一次迭代，故先生成iteration器')
            iter_train = iter(train_loader)
            batch_dt = iter_train.next()
        
        #数据传送
        batch = tuple(t.to(device) for t in batch_dt)
        # print('*'*50)
        # print('batch中有样本个数'+str(len(batch)))
        # print(len(batch))
        raw_input_ids,masked_input_ids, input_mask, segment_ids,is_mix, mlm_pred_positions, mlm_true_ids,mlm_weights = batch
        
        ####这里input_ids应该换做mlm_input_ids否则mlm无法有效训练
        mlm_l, nsp_l, l = net(masked_input_ids,input_mask,segment_ids,mlm_pred_positions,mlm_true_ids,mlm_weights,is_mix)

        #更新，参考：https://zhuanlan.zhihu.com/p/445009191
        tr_loss += l.item()#取item是只取值的意思，不返回相关梯度，减少内存消耗。参考https://blog.csdn.net/weixin_42436099/article/details/118091475
        l.backward()
        optimizer.step()
        schedule.step()
        # net.zero_grad()#将模型优化参数的累计梯度清零，和optimizer.zero_grad()相似度较大，存疑
        optimizer.zero_grad()#将需要优化的参数的累计梯度清零，目的在于每次更新的都是基于这个batch的梯度
        # metric.add(mlm_l, nsp_l, input_ids.shape[0], 1)#tokens_X.shape[0]样本数，即batch_size
        # timer.stop()
        # animator.add(step + 1,(metric[0] / metric[3], metric[1] / metric[3]))
        # step += 1
        # if step == num_steps:
        #     num_steps_reached = True
        #     break
        # print(step)
    
    # print(f'MLM loss {metric[0] / metric[3]:.3f}, 'f'NSP loss {metric[1] / metric[3]:.3f}')
    # print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on 'f'{str(device)}')
    return net,tr_loss

def main(args):
    set_seeds()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PostTrainingModel(args, device).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_dataset = PostTraining(args.max_seq_length, tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                 sampler=RandomSampler(train_dataset),#每次都在采样 
                                 batch_size=args.train_batch_size,
                                 num_workers=8)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler是对optimizer学习率的适时调整
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", args.num_steps)
    # logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Learning rate = %d", args.learning_rate)

    logger.info(" Dataloader的尺寸为len:%d",len(train_dataloader))


    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    model.train()#开启训练模式
    # print('开启训练模式')

    model,tr_loss = post_train(train_loader=train_dataloader,
            net=model,
            device=device,
            num_steps=args.num_steps,
            optimizer=optimizer,
            schedule=scheduler)
    
    #训练完出存
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    print(WEIGHTS_NAME)
    print(CONFIG_NAME)
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    # model_to_save.args.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)
    logger.info(f'train loss = {tr_loss / args.num_steps}')
    # logger.info(f'global steps = {args.num_steps}')
    logger.info("=========== Saving fine - tuned model ===========")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    print('开启参数设置')

    ## Required parameters
    parser.add_argument("--train_corpus",
                        default='../data/',
                        type=str,
                        help="The input train corpus. sports or tv")
    parser.add_argument("--mix_domain",
                        default='../data/',
                        type=str,
                        help="The input train corpus. sports or tv")
    parser.add_argument("--bert_model", default='../PLMs/bert_base_chinese', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=f'./checkpoints/post_{TIME_STAMP}/',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--num_steps",
                        default=20000,
                        type=int,
                        help="Total number of training batch times to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                                "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',#设置了一个开关https://blog.csdn.net/qq_31347869/article/details/104837836
                        default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_steps", default=10000, type=int, help="Maximum steps size")
    parser.add_argument("--hidden_size", default=768, type=int, help="Hidden Vector size")



    args = parser.parse_args(args=[]) #notbook中要加args=[]

    print('完成参数设置')
    print('开始运行主函数')
    main(args)

