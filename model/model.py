import torch.nn as nn
from functions import ReverseLayerF

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction,AdamW, get_linear_schedule_with_warmup
from transformers import BertModel

class BERT_DANN_model(nn.Module):

  def __init__(self,num_inputs,num_hiddens,MODEL_PATH,model_config):
      super(BERT_DANN_model,self).__init__()
      self.bert = BertModel.from_pretrained(MODEL_PATH, config = model_config) # bert预训练模型

      self.class_classifier = nn.Sequential(
                          nn.Linear(num_inputs, num_hiddens),#中间把[CLS]的嵌入做了一次变化，维度从num_inputs变为了num_hiddens
                          nn.Tanh(),
                          nn.Linear(num_hiddens, 2))#对中间层做了Tanh变化后才转到2分类问题

      self.domain_classifier = nn.Sequential(
                          nn.Linear(num_inputs, num_hiddens),
                          nn.Tanh(),
                          nn.Linear(num_hiddens, 2))

      #两个classifier只输出了最后的值，没有进一步计算softmax和交叉熵是因为main函数中nn.CrossEntropyLoss()做了统一化
      #其实nn.CrossEntropyLoss()整合了nn.logsoftmax()和nn.NLLLoss()两部，可以详见
      #https://zhuanlan.zhihu.com/p/159477597
  def forward(self,input_ids,attention_mask,token_type_ids,alpha):
      output = self.bert(
        input_ids,
        attention_mask,
        token_type_ids
      ) 
      pooler_output = output['pooler_output']
      reverse_pooler_output = ReverseLayerF.apply(pooler_output,alpha)
      class_output = self.class_classifier(pooler_output)
      domain_output = self.domain_classifier(reverse_pooler_output)
      return class_output,domain_output


class BERT_raw_classifier(nn.Module):
  '''
    不做posting-training，也不做对抗训练，仅仅是在源域上训练情感分类器，后直接套到目标域上做预测
  '''
  def __init__(self,num_inputs,num_hiddens,MODEL_PATH,model_config):
    '''
      num_inputs:即为BERT模型输出的每个token向量的维度，这里其实只用了[CLS]向量
    '''
    super(BERT_DANN_model,self).__init__()
    self.bert = BertModel.from_pretrained(MODEL_PATH, config = model_config) # bert预训练模型

    self.class_classifier = nn.Sequential(
                        nn.Linear(num_inputs, num_hiddens),#中间把[CLS]的嵌入做了一次变化，维度从num_inputs变为了num_hiddens
                        nn.Tanh(),
                        nn.Linear(num_hiddens, 2))#对中间层做了Tanh变化后才转到2分类问题
  def forward(self,input_ids,attention_mask,token_type_ids):
    output = self.bert(
          input_ids,
          attention_mask,
          token_type_ids
        ) 
    pooler_output = output['pooler_output']
    class_output = self.class_classifier(pooler_output)
    return class_output
  
