# engine.py

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import device, config

def count_trainable_parameters(model):
    """计算模型中可训练参数的数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, X, Y, epochs, lr):
    """训练模型的函数。"""
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    X, Y = X.to(device), Y.to(device)
    
    losses = []
    
    print("开始指令微调...")
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = 0
        num_batches = 0
        for i in range(0, len(X), 4): # 假设 batch_size=4
            xb = X[i:i+4]
            yb = Y[i:i+4]
            _, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        losses.append(epoch_loss / num_batches)
        
    print("训练完成!")
    return losses

@torch.no_grad()
def generate_greedy(model, instruction, input_text, tokenizer, max_new_tokens=150):
    """使用贪心策略生成文本。"""
    from data_utils import format_prompt
    model.eval()
    prompt = format_prompt({"instruction": instruction, "input": input_text})
    prompt_ids = tokenizer.tokenize(prompt)
    x = torch.LongTensor([prompt_ids]).to(device)
    eos_token_id = tokenizer.word_to_idx['<eos>']
    
    for _ in range(max_new_tokens):
        x_cond = x[:, -config.max_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        if idx_next.item() == eos_token_id: break
        x = torch.cat((x, idx_next), dim=1)

    generated_ids = x[0][len(prompt_ids):].tolist()
    return tokenizer.detokenize(generated_ids)

@torch.no_grad()
def generate_sampling(model, instruction, input_text, tokenizer, max_new_tokens=150, temperature=1.0):
    """使用采样策略生成文本。"""
    from data_utils import format_prompt
    model.eval()
    prompt = format_prompt({"instruction": instruction, "input": input_text})
    prompt_ids = tokenizer.tokenize(prompt)
    x = torch.LongTensor([prompt_ids]).to(device)
    eos_token_id = tokenizer.word_to_idx['<eos>']

    for _ in range(max_new_tokens):
        x_cond = x[:, -config.max_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == eos_token_id: break
        x = torch.cat((x, idx_next), dim=1)

    generated_ids = x[0][len(prompt_ids):].tolist()
    return tokenizer.detokenize(generated_ids)

def evaluate_and_get_metrics(model, eval_data, tokenizer, generation_fn):
    """评估模型并返回BLEU和ROUGE分数。"""
    model.to(device)
    model.eval()
    
    bleu_scores = []
    rouge_l_f1_scores = []
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    print(f"\n正在对 {len(eval_data)} 条样本进行评估...")
    for item in tqdm(eval_data, desc="Evaluating"):
        instruction = item['instruction']
        input_text = item['input']
        reference = item['output']
        
        generated_text = generation_fn(model, instruction, input_text, tokenizer)
        
        ref_tokens = list(reference)
        gen_tokens = list(generated_text)
        if len(gen_tokens) > 0:
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)
        
        scores = scorer.score(reference, generated_text)
        rouge_l_f1_scores.append(scores['rougeL'].fmeasure)

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge_l = np.mean(rouge_l_f1_scores) if rouge_l_f1_scores else 0
    
    print(f"评估完成: Avg BLEU = {avg_bleu:.4f}, Avg ROUGE-L = {avg_rouge_l:.4f}")
    return avg_bleu, avg_rouge_l