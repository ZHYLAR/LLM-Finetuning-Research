import json
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List # 导入List以兼容旧版Python

# --- 1. 配置 ---
# a) 远程API服务器的配置
SERVER_IP = "127.0.0.1"  # 请修改为您的服务器IP地址
SERVER_PORT = 8081
API_URL = f"http://{SERVER_IP}:{SERVER_PORT}/v1/chat/completions"

# b) 本地RAG检索系统的配置
EMBEDDING_MODEL_PATH = '/data1/zhy/LLaMA-Factory/rag_model'
# c) 修改点 1：将知识库文件路径改为 .json
KNOWLEDGE_BASE_PATH = "/data1/zhy/LLaMA-Factory/data/knowledge_base.json"

# --- 2. 本地检索系统实现 (RAG的'R'部分) ---
class RetrievalSystem:
    """
    在本地运行的检索系统，负责从外部JSON文件加载知识并创建索引。
    """
    def __init__(self, embedding_model_path: str, knowledge_base_path: str):
        print("正在本地加载向量嵌入模型 (首次运行可能需要几分钟)...")
        self.encoder = SentenceTransformer(embedding_model_path)
        self._build_knowledge_base(knowledge_base_path)
        print("向量嵌入模型及本地知识库已准备就绪。")

    # --- 修改点 2：_build_knowledge_base 方法改为解析JSON ---
    def _build_knowledge_base(self, file_path: str):
        print(f"正在从JSON文件 '{file_path}' 读取知识库...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f) # 使用 json.load 读取文件
                # 从每个对象中提取 "content" 键的值，并过滤掉空内容
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

# --- 3. Prompt 模板 (RAG的'A'部分) ---
# ... (这部分代码保持不变) ...
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

# --- 4. 主函数：整合RAG流程并调用API ---
# ... (这部分代码保持不变) ...
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

# --- 5. 程序主入口 ---
if __name__ == "__main__":
    # --- 修改点 3：初始化时传入JSON文件路径 ---
    local_retriever = RetrievalSystem(
        embedding_model_path=EMBEDDING_MODEL_PATH,
        knowledge_base_path=KNOWLEDGE_BASE_PATH
    )
    
    # 创建一个交互式命令行，方便连续提问
    print("\n--- 鞋类知识RAG助手已启动 ---")
    print("现在，您可以开始提问了。例如：'一字拖和人字拖有什么区别？'")
    while True:
        user_question = input("\n请输入你的问题 (输入 'exit' 或 '退出' 来结束): \n> ")
        if user_question.lower() in ['exit', '退出']:
            print("感谢使用，再见！")
            break
        
        final_answer = perform_rag_query(user_question, local_retriever)
        
        print("\n[最终答案]")
        print("="*40)
        print(final_answer)
        print("="*40)
