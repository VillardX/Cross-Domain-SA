import copy
import torch 
from torch.utils.data import Dataset, DataLoader

class GetLoader(Dataset):
    def __init__(self, dt, tokenizer, max_len = 256):
        '''
            输入：
                dt:dataframe，两列review，is_positive
        '''
        self.dt = dt
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, idx):
        sentiment_label = self.dt['is_positive'][idx].tolist()
        encoding = self.tokenizer(
            self.dt['review'][idx],
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,        # 添加特殊tokens 【cls】【sep】
            return_token_type_ids=True,    # 返回是前一句还是后一句
            return_attention_mask=True,     # 返回attention_mask
            return_tensors='pt',             # 返回pytorch类型的tensor
            truncation=True # 若大于max则切断
        )
        return {
            'sentiment_label':torch.tensor(sentiment_label,dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

    def __len__(self):
        return self.dt.shape[0]
