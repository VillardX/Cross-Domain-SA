import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets


def test(dataset_name,tokenizer,max_len=256):
    '''
        dataset_name in ['game', 'meituan']
    '''
    assert dataset_name in ['game', 'meituan']

    model_root = '../temp_model_output'
    data_root = '../data'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cuda = True
    # cudnn.benchmark = True
    batch_size = 32
    alpha = 0

    """load data"""
    if dataset_name == 'game':#说明测试的是game的数据集
        test_list = os.path.join(data_root, 'game_test1.csv')#以后这里需要模块化
        dt = pd.read_csv(test_list,encoding='utf-8')
        dt['review'] = dt['review'].astype(str)#加一行
        dataset = GetLoader(
            dt,
            tokenizer,
            max_len
        )
    else:
        test_list = os.path.join(data_root, 'meituan_test1.csv')#以后这里需要模块化
        dt = pd.read_csv(test_list,encoding='utf-8')
        dt['review'] = dt['review'].astype(str)#加一行
        dataset = GetLoader(
            dt,
            tokenizer,
            max_len
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'game2meituan_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    my_net = my_net.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()

        t_sentiment_label = data_target['sentiment_label'].to(device)
        t_input_ids = data_target['input_ids'].to(device)
        t_attention_mask = data_target['attention_mask'].to(device)
        t_token_type_ids = data_target['token_type_ids'].to(device)

        batch_size = len(t_sentiment_label)

        class_output, _ = my_net(t_input_ids,t_attention_mask,t_token_type_ids,alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_sentiment_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
