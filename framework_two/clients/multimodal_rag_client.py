import json
import faiss
import requests
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from typing import List

# --- 1. 全局配置 ---
# a) 远程API服务器的配置
SERVER_IP = "127.0.0.1"  # 请修改为您的服务器IP地址
SERVER_PORT = 8081
API_URL = f"http://{SERVER_IP}:{SERVER_PORT}/v1/chat/completions"

# b) 本地RAG检索系统的配置
EMBEDDING_MODEL_PATH = '/data1/zhy/LLaMA-Factory/rag_model'
KNOWLEDGE_BASE_PATH = "/data1/zhy/LLaMA-Factory/data/knowledge_base.json"

# c) 本地CLIP图像处理系统的配置
CLIP_MODEL_PATH = "/data1/zhy/LLaMA-Factory/CLIP"
TAGS_FILE_PATH = '/data1/zhy/LLaMA-Factory/CLIP/sneaker_tag.txt'

# --- 2. 图片处理系统 (CLIP) ---
class ImageTagger:
    """
    使用CLIP模型处理图片，提取特征标签并生成一个文本Prompt。
    """
    def __init__(self, model_path: str, tags_path: str):
        print("正在加载CLIP模型及处理器...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CLIP模型将使用设备: {self.device}")
        
        try:
            self.model = CLIPModel.from_pretrained(model_path).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        except Exception as e:
            print(f"[错误] 加载CLIP模型失败: {e}")
            raise
            
        self._load_and_encode_tags(tags_path)
        print("CLIP模型及标签库已准备就绪。")

    def _load_and_encode_tags(self, tags_path: str):
        print(f"正在从 '{tags_path}' 加载候选标签...")
        try:
            with open(tags_path, 'r', encoding='utf-8') as f:
                self.candidate_tags = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            print(f"加载了 {len(self.candidate_tags)} 个候选标签。")
        except FileNotFoundError:
            print(f"[错误] 标签文件 '{tags_path}' 未找到。")
            self.candidate_tags = []

        if not self.candidate_tags:
            print("[警告] 候选标签列表为空，图片识别功能将受限。")
            return

        print("正在预编码所有候选标签以提高效率...")
        with torch.no_grad():
            text_inputs = self.processor(text=self.candidate_tags, return_tensors="pt", padding=True, truncation=True).to(self.device)
            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def generate_prompt_from_image(self, image_path: str, top_k: int = 3, question_suffix: str = "") -> str:
        if not self.candidate_tags:
            return "错误：由于候选标签列表为空，无法处理图片。"
            
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return f"错误：图片文件 '{image_path}' 未找到。"
        except Exception as e:
            return f"错误：打开图片时发生未知错误: {e}"
        
        with torch.no_grad():
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_feature = self.model.get_image_features(**image_inputs)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)

        similarity_scores = (image_feature @ self.text_features.T).squeeze(0)
        top_k_indices = similarity_scores.topk(top_k).indices
        best_tags = [self.candidate_tags[i] for i in top_k_indices]
        
        print(f"-> 图片识别出的最佳标签: {', '.join(best_tags)}")
        tags_string = "、".join(best_tags)
        
        # 将识别出的标签和用户问题融合
        base_prompt = f"关于这双看起来具有“{tags_string}”特征的鞋子"
        if question_suffix:
             final_prompt = f"{base_prompt}，请回答我的问题：“{question_suffix}”"
        else:
             final_prompt = f"请详细介绍一下这双具有“{tags_string}”特征的鞋子。"

        return final_prompt

# --- 3. 文本检索系统 (RAG的'R'部分) ---
class RetrievalSystem:
    """
    在本地运行的检索系统，负责从外部JSON文件加载知识并创建索引。
    """
    def __init__(self, embedding_model_path: str, knowledge_base_path: str):
        print("正在本地加载向量嵌入模型 (首次运行可能需要几分钟)...")
        try:
            self.encoder = SentenceTransformer(embedding_model_path)
        except Exception as e:
            print(f"[错误] 加载SentenceTransformer模型失败: {e}")
            raise
        self._build_knowledge_base(knowledge_base_path)
        print("向量嵌入模型及本地知识库已准备就绪。")

    def _build_knowledge_base(self, file_path: str):
        print(f"正在从JSON文件 '{file_path}' 读取知识库...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
                self.text_chunks = [
                    item['content'].strip() for item in knowledge_data if 'content' in item and item['content'].strip()
                ]
        except FileNotFoundError:
            print(f"[错误] 知识库文件 '{file_path}' 未找到。请确保文件存在于正确的位置。")
            self.text_chunks = []
        except json.JSONDecodeError:
            print(f"[错误] 知识库文件 '{file_path}' 格式不正确，无法解析JSON。")
            self.text_chunks = []
        except KeyError:
            print(f"[错误] 知识库文件 '{file_path}' 中的部分条目缺少 'content' 键。")
            self.text_chunks = []

        if not self.text_chunks:
            print("[警告] 知识库为空或加载失败。检索功能将无法返回任何结果。")
            return

        print(f"知识库加载成功，共 {len(self.text_chunks)} 条知识。正在创建向量索引...")
        vectors = self.encoder.encode(self.text_chunks, normalize_embeddings=True)
        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap(self.index)
        ids = np.arange(len(self.text_chunks)).astype('int64')
        self.index.add_with_ids(vectors, ids)
        print("向量索引创建完毕。")

    def search(self, query: str, k: int = 2) -> List[str]:
        if not hasattr(self, 'index'):
            return []
        query_vector = self.encoder.encode([query], normalize_embeddings=True)
        distances, retrieved_ids = self.index.search(query_vector, k)
        return [self.text_chunks[i] for i in retrieved_ids[0]]

# --- 4. Prompt 模板 (RAG的'A'部分) ---
PROMPT_TEMPLATE = """
你是一个专业的、风格口语化的鞋类导购。请根据下面提供的、各自独立的上下文信息，来友好地回答用户的问题。
你的回答需要满足以下要求：
1.  **综合并优先使用**我提供的“上下文信息”来组织你的回答。
2.  如果上下文信息与问题无关或不足以回答，请礼貌地说明“根据我手头的资料，无法回答这个问题”，然后再根据你自己的知识尝试回答。
3.  保持你在微调后学到的那种亲切、自然的对话风格。

--- 以下是独立的参考资料 ---
{context}
--- 参考资料结束 ---

请根据以上资料，回答我的问题：{question}
"""

# --- 5. RAG查询与API调用 ---
def perform_rag_query(question: str, retriever: RetrievalSystem):
    print("\n[步骤 1/3] 正在本地知识库中检索相关信息...")
    retrieved_chunks = retriever.search(question)
    formatted_context = ""
    if not retrieved_chunks:
        formatted_context = "（未在本地知识库中找到相关信息）"
    else:
        print(f"-> 检索到 {len(retrieved_chunks)} 条上下文信息。")
        for i, chunk in enumerate(retrieved_chunks):
            formatted_context += f"--- 参考资料 {i+1} ---\n"
            formatted_context += chunk + "\n\n"
    print(f"-> 格式化后的上下文:\n{formatted_context}")
    print("\n[步骤 2/3] 正在本地构建最终的Prompt...")
    final_prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=question)
    print(f"-> 构建完成的Prompt，即将发送到API。")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3-dpo",
        "messages": [{"role": "user", "content": final_prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    print("\n[步骤 3/3] 正在向远程API服务器发送请求...")
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        response_data = response.json()
        answer = response_data['choices'][0]['message']['content']
        print("-> 成功从服务器获取回答！")
        return answer
    except requests.exceptions.RequestException as e:
        error_message = f"请求 API 时出错: {e}"
        print(f"-> {error_message}")
        return error_message

# --- 6. 程序主入口 (多模态交互) ---
if __name__ == "__main__":
    try:
        # 初始化两个核心系统
        local_retriever = RetrievalSystem(
            embedding_model_path=EMBEDDING_MODEL_PATH,
            knowledge_base_path=KNOWLEDGE_BASE_PATH
        )
        image_tagger = ImageTagger(
            model_path=CLIP_MODEL_PATH,
            tags_path=TAGS_FILE_PATH
        )
    except Exception as initialization_error:
        print(f"\n系统初始化失败，程序无法启动: {initialization_error}")
        exit()

    print("\n" + "="*50)
  
    print("您可以在提问后，选择性地提供图片路径以获得更精准的回答。")
    print("="*50)
    
    while True:
        # --- 修改点：重构交互逻辑 ---
        
        # 1. 首先，获取用户的文字问题
        user_question = input("\n请输入您的问题 (输入 'exit' 退出):\n> ")
        
        if user_question.lower() in ['exit', '退出']:
            print("感谢使用，再见！")
            break
            
        if not user_question.strip():
            print("问题不能为空，请重新输入。")
            continue

        # 2. 然后，获取可选的图片路径
        image_path = input("请提供相关的图片路径 (可选, 直接按回车跳过):\n> ").strip()

        question_to_ask = None
        
        # 3. 根据有无图片路径，决定最终的问题
        if image_path:
            # 如果有图片，则融合图片信息和文字问题
            print("\n[信息融合] 正在处理图片并结合您的问题...")
            question_to_ask = image_tagger.generate_prompt_from_image(
                image_path, 
                question_suffix=user_question
            )
            
            if question_to_ask.startswith("错误："):
                print(f"\n[处理失败] {question_to_ask}")
                continue # 图片处理失败，跳过本次查询
            
            print(f"-> 融合后的查询是:\n---\n{question_to_ask}\n---")
        else:
            # 没有图片，直接使用用户的原始问题
            question_to_ask = user_question

        # --- 统一执行RAG查询 ---
        if question_to_ask:
            final_answer = perform_rag_query(question_to_ask, local_retriever)
            
         
            print("="*40)
            print(final_answer)
            print("="*40)
