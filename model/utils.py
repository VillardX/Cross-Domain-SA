import os
import logging
import logging.config
import yaml
from datetime import datetime
import torch
import random
from transformers import AutoTokenizer

TIME_STAMP = datetime.now().strftime('%Y-%m-%dT%H:%M')

import re

def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_name_or_path)

def init_logger():
    if not os.path.exists(f'post-training/checkpoints/'):
        os.mkdir(f'post-training/checkpoints/')

    if not os.path.exists(f'post-training/checkpoints/{TIME_STAMP}/'):
        os.mkdir(f'post-training/checkpoints/{TIME_STAMP}/')
    
    config = yaml.load(open('./logger.yml'), Loader=yaml.FullLoader)
    config['handlers']['file_info']['filename'] = f'post-training/checkpoints/{TIME_STAMP}/train.log'
    logging.config.dictConfig(config)

def set_seeds():
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds)).argmax(axis=1)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return rounded_preds, acc

def format_time(end, start):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
