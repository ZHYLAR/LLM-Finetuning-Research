# 框架二：服务化领域专家模型

## 1. 项目简介
该项目完整地展示了如何基于 **[Llama Factory框架](https://github.com/hiyouga/LLaMA-Factory)**，通过 **SFT (监督微调)** 和 **DPO (直接偏好优化)** 两阶段训练，将一个通用大语言模型（如 Llama-3-8B）改造为特定领域（以“鞋类”为例）的专家模型。

此外，项目还包括了如何将训练好的模型通过 **API 服务化**，并构建**检索增强生成 (RAG)** 和 **多模态（视觉辅助）** 的下游应用客户端，形成一个端到端的、从模型训练到应用部署的完整解决方案。

## 2. 实验架构
本项目采用解耦的服务化架构，模拟了真实的产业应用场景。
- **A100 服务器（模型服务商）**: 负责模型的微调、推理，并通过 API 提供核心语言能力。
- **3080ti 服务器（下游任务厂商）**: 运行 RAG 和多模态客户端，负责整合业务逻辑（如知识库检索）并调用模型 API。
- 个人电脑（用户）: 模拟最终用户，通过 SSH 端口映射与应用交互。

## 3. 文件结构

```
/framework_two/
├── scripts/
│   ├── run_sft.sh                  \# 监督微调 (SFT) 脚本
│   ├── run_dpo.sh                  \# 直接偏好优化 (DPO) 脚本
│   └── start_api.sh                \# 启动模型 API 服务脚本
│
├── clients/
│   ├── rag_client.py               \# 纯文本 RAG 客户端
│   └── multimodal_rag_client.py    \# 视觉辅助的多模态 RAG 客户端
│
├── data/
│   ├── shoes.jsonl                 \# (需自备) SFT 指令数据集
│   ├── shoes_dpo.jsonl             \# (需自备) DPO 偏好数据集
│   └── knowledge_base.json         \# (需自备) RAG 知识库
│
├── clip_data/
│   └── sneaker_tag.txt             \# (需自备) CLIP 图像候选标签
│
└── README.md                       \# 本文档

````

## 4. 环境与数据准备

### 4.1. 环境配置

- **操作系统**: Ubuntu 20.04.2 
-  **Python**: 3.10 
-  需先配置[Llama Factory框架](https://github.com/hiyouga/LLaMA-Factory)
- **核心库**:
    -  `torch`: 2.7.1 
    -  `llama-factory`: 0.9.4.dev0
    - `faiss-gpu`
    - `sentence-transformers`
    - `transformers`
    - `Pillow`
    - `requests`

-  **硬件**: 实验在 NVIDIA A100 GPU (40G) 上进行。

### 4.2. 预训练模型准备

下载以下模型到本地：
- **LLM 基础模型**: `Meta-Llama-3-8B-Instruct`
- **Embedding 模型**:[`BAAI/bge-large-zh-v1.5`](https://huggingface.co/BAAI/bge-large-zh)
- **CLIP 模型**:[`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)

### 4.3. 数据准备

请根据以下格式准备数据文件：

1.  **SFT 数据 (`data/shoes.jsonl`)**: 用于指令微调的 JSON Lines 文件。
2.   **DPO 数据 (`data/shoes_dpo.jsonl`)**: 用于偏好学习的 JSON Lines 文件，每行包含 "prompt", "chosen", "rejected" 三个键。实验中，“chosen”被设计为比“rejected”更口语化的回答。
3.   **RAG 知识库 (`data/knowledge_base.json`)**: 一个 JSON 文件，包含一个对象列表，每个对象有一个 "content" 键，值为知识条目。
4.   **CLIP 标签 (`clip_data/sneaker_tag.txt`)**: 一个纯文本文件，每行一个候选标签，用于图像识别。

## 5. 复现步骤

### 步骤 1: SFT 监督微调

首先，对基础模型进行 SFT，注入领域知识。

- **运行脚本**: `./scripts/run_sft.sh`
-  **预期结果**: 训练完成后，模型的回答能力将显著提升。实验中，评估指标从基线的 `BLEU-4=0.2765` / `ROUGE-L=2.2573`  提升至 `BLEU-4=19.0503` / `ROUGE-L=33.5724` 。模型从无法理解领域术语（如将“一字拖”解释为书法技巧），变为了能够准确回答相关问题 。

### 步骤 2: DPO 直接偏好优化

在 SFT 模型的基础上，进行 DPO 训练，使模型的回答风格更符合人类偏好（更口语化）。

- **运行脚本**: `./scripts/run_dpo.sh`
-  **预期结果**: DPO 训练后，模型的回答风格会变得更加自然、口语化。实验中，DPO 评估指标 `rewards/accuracies` 达到了 `1.0`，表明模型已能很好地区分偏好数据中的“好”与“坏”回答。

### 步骤 3: 启动 API 服务

将经过 SFT 和 DPO 两阶段训练的模型部署为 API 服务。此脚本会同时加载 SFT 和 DPO 的 LoRA 适配器。

- **运行脚本**: `./scripts/start_api.sh`
-  **说明**: 脚本会在 `0.0.0.0:8000` 启动一个符合 OpenAI 规范的 API 服务。

### 步骤 4: 运行 RAG 客户端

启动 RAG 客户端，该客户端会连接到上述 API 服务，实现基于外部知识库的问答。

- **运行脚本**: `python ./clients/rag_client.py`
-  **使用**: 启动后，根据提示输入您的问题，客户端会先从 `knowledge_base.json` 检索信息，然后将信息和问题一起发送给大模型，并返回最终答案。

### 步骤 5: 运行多模态 RAG 客户端

这是更高级的客户端，在 RAG 的基础上增加了视觉辅助功能。

- **运行脚本**: `python ./clients/multimodal_rag_client.py`
-  **使用**: 启动后，先输入文字问题，然后可以选择性地输入本地图片路径。如果提供了图片，程序会先用 CLIP 模型识别图片特征（如“涂鸦印花,蓝色,复古跑鞋”），将这些特征融入问题，再执行 RAG 流程。

---
