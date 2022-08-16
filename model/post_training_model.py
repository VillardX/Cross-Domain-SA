import numbers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from transformers import AutoModel, AutoConfig,BertTokenizer, BertConfig

class MaskLM(nn.Module):
    """
    The masked language model task of BERT.
    """
    def __init__(self, config, num_hiddens, num_inputs=768, **kwargs):
        '''
        num_hiddens指单层mlp中间层的单元数，num_inputs是输入的维度
        '''
        super(MaskLM, self).__init__(**kwargs)
        vocab_size = config.vocab_size
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                #尺寸从(batch_size,seq_len,num_inputs)变成(batch_size,seq_len,num_hiddens)
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 #对输入的最后一维进行归一化，参数为一个整数，需要等于最后一维的维度
                                 #即对每个embedding进行归一化
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        '''
            此处传入的应该是padded_pred_positions，X为bert输出的last_hidden_state
        '''
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)#拉成一维
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)#重复batch_idx中的每个元素num_pred_positions，并按原序输出
        masked_X = X[batch_idx, pred_positions]#取出每个样本（batch）对应的每个被mask位置的embedding
        #masked_X最后得到的是一个二维矩阵，每一行代表一个mask词的embedding，共有num_masked行
        #有些pred_positions是padding出来的，所以这里会多取许多每个batch的0位置向量，但是后期计算loss的时候会用到mlm_weight的权重为0
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))#由于padding。每个样本的mlm数量是一样的，所以可以直接reshape
        #因为设置了每个batch样本的masked词数量是一样的，所以的可以再将矩阵规整
        #本身取就是按batch顺序取的，所以reshape的时候可以直接排
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """
        The next sentence prediction task of BERT.
    """
    def __init__(self, num_hiddens, num_inputs=768,**kwargs):
        '''
            num_hiddens指单层mlp中间层的单元数，num_inputs是输入的维度
        '''
        super(NextSentencePred, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                    nn.Tanh(),
                                   nn.Linear(num_hiddens, 2))

    def forward(self, X):
        '''
            此处传入的X为bert输出的pooler_output
        '''
        nsp_Y_hat = self.mlp(X)
        return nsp_Y_hat


class PostTrainingModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        config = AutoConfig.from_pretrained(args.bert_model)
        self.config = config

        self.bert = AutoModel.from_pretrained(args.bert_model, config=config)
        self.mlm = MaskLM(config,num_hiddens=768,num_inputs=768)
        self.nsp = NextSentencePred(num_hiddens=768,num_inputs=768)
    
    def forward(self,masked_input_ids,attention_mask,token_type_ids,padded_pred_positions,mlm_true_ids,mlm_weights,is_mix):
        output = self.bert(masked_input_ids,attention_mask,token_type_ids)
        mlm_Y_hat = self.mlm(X=output['last_hidden_state'],pred_positions=padded_pred_positions)
        nsp_Y_hat = self.nsp(X=output['pooler_output'])

        loss = nn.CrossEntropyLoss(reduction='none')#损失函数初始化.to(device)，在训练的时候会把整个网络to(device)
        #参数reduction默认为"mean"，表示对所有样本的loss取均值，最终返回只有一个值
        #参数reduction取"none"，表示保留每一个样本的loss
        #参考https://blog.csdn.net/u012633319/article/details/111093144
        mlm_l, nsp_l, l = _get_batch_loss_bert(loss,mlm_Y_hat,nsp_Y_hat,
                                                mlm_true_ids,mlm_weights,is_mix,
                                                config=self.config)


        return mlm_l, nsp_l, l
        
def _get_batch_loss_bert(loss,
                    mlm_Y_hat,nsp_Y_hat,
                        mlm_Y,mlm_weights_X,nsp_y,
                        config):
    '''
        获取MLM和NSP任务的损失
        输入：
            net：为torch_bert
            loss:nn.crossentropy()函数
            config:PLM的config
            input_ids,attention_mask,token_type_ids,padded_pred_postitions:即为batch_data中的各个键
            mlm_Y_hat,nsp_Y_hat:输出值
            mlm_Y：对应batch_data['padded_mlm_pred_labels']
            mlm_weights_X：对应batch_data['mlm_weights']
            nsp_y:对应batch_data['NSP_target']
    '''
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, config.vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1)#乘以各自权重决定有不有效
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)#求平均
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    nsp_l = nsp_l.mean() 
    #计算总的loss
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
