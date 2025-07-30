# main.py

import torch
import io
import random

# 导入配置
from config import config, device, NUM_EPOCHS, LEARNING_RATE, EVAL_SAMPLE_SIZE, FIGURE_SAVE_PATH, file_path

# 导入模型和数据处理工具
from model import ToyLLM, LoRALayer, MultiHeadAttention
from data_utils import load_data, CharTokenizer, create_dataset

# 导入引擎（训练、评估、生成）
from engine import train, evaluate_and_get_metrics, generate_greedy, generate_sampling, count_trainable_parameters

# 导入绘图工具
from plot_utils import plot_convergence, plot_bar_comparison, plot_radar_chart

def main():
    # --- 数据加载与预处理 ---
    
    finetune_data = load_data(file_path)
    
    finetune_data = finetune_data[:20] # 截取部分数据用于快速演示
    
    tokenizer = CharTokenizer(finetune_data)
    finetune_X, finetune_Y = create_dataset(finetune_data, tokenizer, config.max_len)
    
    random.seed(42)
    evaluation_subset = random.sample(finetune_data, EVAL_SAMPLE_SIZE)

    # --- 保存基础模型状态，用于后续加载 ---
    base_model = ToyLLM(config)
    buffer = io.BytesIO()
    torch.save(base_model.state_dict(), buffer)
    
    results = {}

    # --- 方案 1 & 2: LoRA 微调 ---
    print("\n" + "="*50 + "\n方案 1 & 2: LoRA 微调\n" + "="*50)
    
    buffer.seek(0) # 重置缓冲区指针
    lora_model = ToyLLM(config)
    lora_model.load_state_dict(torch.load(buffer, map_location='cpu'))
    
    for param in lora_model.parameters():
        param.requires_grad = False
    
    lora_rank, lora_alpha = 16, 32
    for name, module in lora_model.named_modules():
        if isinstance(module, MultiHeadAttention):
            module.q_proj = LoRALayer(module.q_proj, rank=lora_rank, alpha=lora_alpha)
            module.v_proj = LoRALayer(module.v_proj, rank=lora_rank, alpha=lora_alpha)
            
    lora_model.to(device)
    lora_params = count_trainable_parameters(lora_model)
    print(f"\nLoRA 模型可训练参数: {lora_params}")

    lora_losses = train(lora_model, finetune_X, finetune_Y, epochs=NUM_EPOCHS, lr=LEARNING_RATE)
    bleu_lg, rouge_lg = evaluate_and_get_metrics(lora_model, evaluation_subset, tokenizer, generate_greedy)
    results['LoRA + Greedy'] = {'params': lora_params, 'bleu': bleu_lg, 'rouge': rouge_lg}
    bleu_ls, rouge_ls = evaluate_and_get_metrics(lora_model, evaluation_subset, tokenizer, generate_sampling)
    results['LoRA + Sampling'] = {'params': lora_params, 'bleu': bleu_ls, 'rouge': rouge_ls}

    # --- 方案 3 & 4: 全量微调 ---
    print("\n" + "="*50 + "\n方案 3 & 4: 全量微调\n" + "="*50)
    
    buffer.seek(0) # 再次重置缓冲区指针
    full_model = ToyLLM(config)
    full_model.load_state_dict(torch.load(buffer, map_location='cpu'))
    full_model.to(device)
    
    full_params = count_trainable_parameters(full_model)
    print(f"\n全量模型可训练参数: {full_params}")

    full_losses = train(full_model, finetune_X, finetune_Y, epochs=NUM_EPOCHS, lr=LEARNING_RATE)
    bleu_fg, rouge_fg = evaluate_and_get_metrics(full_model, evaluation_subset, tokenizer, generate_greedy)
    results['Full + Greedy'] = {'params': full_params, 'bleu': bleu_fg, 'rouge': rouge_fg}
    bleu_fs, rouge_fs = evaluate_and_get_metrics(full_model, evaluation_subset, tokenizer, generate_sampling)
    results['Full + Sampling'] = {'params': full_params, 'bleu': bleu_fs, 'rouge': rouge_fs}

    # --- 绘图 ---
    print("\n" + "="*50 + "\n正在生成结果图表...\n" + "="*50)
    plot_convergence(lora_losses, full_losses, FIGURE_SAVE_PATH)
    plot_bar_comparison(results, FIGURE_SAVE_PATH)
    plot_radar_chart(results, FIGURE_SAVE_PATH)

    # --- 打印两条验证集回答 ---
    print("\n" + "="*50 + "\n生成回答对比\n" + "="*50)
    validation_questions = [
        {"instruction": "一字拖是什么？", "input": "一字拖"},
        {"instruction": "一字拖适合什么运动？", "input": "一字拖"}
    ]
    models_to_test = {
        "LoRA + Greedy": (lora_model, generate_greedy),
        "LoRA + Sampling": (lora_model, generate_sampling),
        "Full + Greedy": (full_model, generate_greedy),
        "Full + Sampling": (full_model, generate_sampling),
    }

    for q_item in validation_questions:
        print(f"\n--- 指令: {q_item['instruction']} ---")
        for name, (model, gen_fn) in models_to_test.items():
            answer = gen_fn(model, q_item['instruction'], q_item['input'], tokenizer)
            print(f"[{name}]\n回答: {answer}\n")

if __name__ == '__main__':
    main()