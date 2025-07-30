# config.py

import torch
import os

# ==============================================================================
# --- 1. 全局参数设置 ---
# ==============================================================================
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
EVAL_SAMPLE_SIZE = 10
FIGURE_SAVE_PATH = '/data1/zhy/TOY_LLM/com_exp'
file_path = '/data1/zhy/TOY_LLM/lora_train.json'
# 确保保存图片的目录存在
os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)


# ==============================================================================
# --- 2. 模型与环境配置 ---
# ==============================================================================
# 检测并设置设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ModelConfig:
    """
    存放模型的核心架构参数
    """
    vocab_size = 500  # 初始值，会被Tokenizer动态更新
    d_model = 512
    max_len = 256
    num_layers = 36
    n_head = 8
    d_ff = 512
    dropout = 0.1

# 实例化配置类
config = ModelConfig()