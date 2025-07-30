# data_utils.py

import json
import torch
from config import config

class CharTokenizer:
    """
    基于字符的分词器。
    """
    def __init__(self, data):
        chars = set()
        for item in data:
            chars.update(item['instruction'])
            chars.update(item['input'])
            chars.update(item['output'])
            
        self.special_tokens = ['<pad>', '<eos>', '<unk>']
        self.vocab = self.special_tokens + sorted(list(chars))
        self.word_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_word = {i: ch for i, ch in enumerate(self.vocab)}
        
        # 动态更新全局配置中的词汇表大小
        config.vocab_size = len(self.vocab)
        print(f"词汇表大小已更新: {config.vocab_size}")

    def tokenize(self, text):
        return [self.word_to_idx.get(char, self.word_to_idx['<unk>']) for char in text]

    def detokenize(self, ids):
        return "".join([self.idx_to_word.get(i, '<unk>') for i in ids])

def load_data(file_path):
    """从JSON文件加载数据。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 无法找到数据文件 {file_path}")
        exit()

def format_prompt(item):
    """格式化输入，构建prompt。"""
    return f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 回答:\n"

def create_dataset(data, tokenizer, max_len):
    """根据原始数据、分词器和最大长度创建输入和目标张量。"""
    inputs, targets = [], []
    eos_token_id = tokenizer.word_to_idx['<eos>']
    pad_token_id = tokenizer.word_to_idx['<pad>']
    
    for item in data:
        prompt = format_prompt(item)
        response = item['output']
        
        prompt_ids = tokenizer.tokenize(prompt)
        response_ids = tokenizer.tokenize(response)
        
        full_ids = prompt_ids + response_ids + [eos_token_id]
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
            
        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]
        
        prompt_len = len(prompt_ids)
        masked_targets = [-1] * prompt_len + target_ids[prompt_len:]
        
        pad_len = max_len - len(input_ids)
        input_ids += [pad_token_id] * pad_len
        masked_targets += [-1] * pad_len
        
        inputs.append(input_ids)
        targets.append(masked_targets)
        
    return torch.LongTensor(inputs), torch.LongTensor(targets)