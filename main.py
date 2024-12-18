import warnings
import logging
import torch
import pickle
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from haystack.nodes import FARMReader
from haystack import Document
import faiss
import os
import time

# 抑制所有警告
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ['HF_HOME'] = 'E:\\huggingface\\transformers'
os.environ['TRANSFORMERS_CACHE'] = 'E:\\huggingface\\transformers'

# 配置日志只显示错误信息
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 禁用 transformers 的警告
logging.getLogger("transformers").setLevel(logging.ERROR)

class QASystem:
    def __init__(self, embedding_path):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 加载保存的向量和文档内容
            with open(embedding_path, 'rb') as f:
                data = pickle.load(f)
            self.document_embedding = data['embedding']
            self.document_content = data['content']
            
            # 初始化FAISS索引
            self.index = faiss.IndexFlatL2(self.document_embedding.shape[1])
            self.index.add(self.document_embedding.numpy())
            
            # 加载问答模型（静默模式）
            print("正在加载模型，请稍候...")
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base",
                cache_dir='E:\\huggingface\\transformers',
                local_files_only=True
            )
            self.model = DPRQuestionEncoder.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base",
                cache_dir='E:\\huggingface\\transformers',
                local_files_only=True
            ).to(self.device)
            
            # 优化 FARMReader 配置
            self.reader = FARMReader(
                model_name_or_path="deepset/roberta-base-squad2",
                use_gpu=torch.cuda.is_available(),
                top_k=1,
                max_seq_len=384,
                doc_stride=128,
                max_query_length=64,
                return_no_answer=False,
                progress_bar=False,
                num_processes=1
            )
            print("模型加载完成！\n")
            
            # 添加答案缓存
            self.answer_cache = {}
            
            # 预处理文档
            self.processed_documents = self._preprocess_document()
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise

    def _preprocess_document(self):
        """预处理文档，将文档分成小段"""
        paragraphs = self.document_content.split('\n')
        documents = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                doc = Document(
                    content=para.strip(),
                    id=str(i),
                    meta={"name": f"段落_{i}"}
                )
                documents.append(doc)
        return documents

    def get_context(self, answer_text, window_size=200):
        """获取答案的上下文"""
        try:
            # 在文档中找到答案的位置
            pos = self.document_content.find(answer_text)
            if pos == -1:
                return answer_text
            
            # 获取答案前后的文本
            start = max(0, pos - window_size)
            end = min(len(self.document_content), pos + len(answer_text) + window_size)
            
            # 调整到完整句子
            while start > 0 and self.document_content[start] not in '。！？\n':
                start -= 1
            while end < len(self.document_content) and self.document_content[end] not in '。！？\n':
                end += 1
            
            context = self.document_content[start:end].strip()
            return context
        except:
            return answer_text

    def answer_question(self, question):
        if question in self.answer_cache:
            logger.info("从缓存中获取答案")
            return self.answer_cache[question]
        
        logger.info(f"处理问题: {question}")
        print("正在分析文档寻找答案，请稍候...")
        
        try:
            print("1/4 正在处理问题...")
            # 问题向量化
            with torch.no_grad():
                inputs = self.tokenizer(
                    question, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(self.device)
                question_embedding = self.model(**inputs).pooler_output.cpu()
            
            print("2/4 正在检索相关内容...")
            # 向量检索
            D, I = self.index.search(question_embedding.numpy(), k=3)  # 检索前3个相关段落
            
            print("3/4 正在分析答案...")
            # 在相关段落中查找答案
            relevant_docs = [self.processed_documents[i] for i in I[0] if i < len(self.processed_documents)]
            
            # 生成答案
            result = self.reader.predict(
                query=question,
                documents=relevant_docs
            )
            
            print("4/4 正在整理结果...")
            if result['answers'] and len(result['answers']) > 0:
                answer_text = result['answers'][0].answer
                context = self.get_context(answer_text)
                answer = {
                    'short_answer': answer_text,
                    'context': context,
                    'confidence': result['answers'][0].score
                }
            else:
                answer = {
                    'short_answer': "抱歉，我没有找到相关答案",
                    'context': "",
                    'confidence': 0.0
                }
            
            # 存入缓存
            self.answer_cache[question] = answer
            return answer
            
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            return {
                'short_answer': "处理过程出错，请重试",
                'context': "",
                'confidence': 0.0
            }

    def interactive_mode(self):
        def clear_screen():
            os.system('cls' if os.name == 'nt' else 'clear')
            print_welcome_message()
        
        def print_welcome_message():
            print("\n欢迎使用医疗保险政策问答系统！")
            print("=" * 50)
            print("命令说明：")
            print("- 输入 'q' 或 'quit' 退出系统")
            print("- 输入 'clear' 清屏")
            print("- 输入 'cache' 查看已缓存的问题")
            print("=" * 50)
        
        print_welcome_message()
        
        while True:
            try:
                question = input("\n>>> ").strip()
                
                # 跳过空输入和警告消息
                if not question or "warning" in question.lower() or "future" in question.lower():
                    continue
                
                # 处理特殊命令
                if question.lower() in ['q', 'quit']:
                    print("\n感谢使用，再见！")
                    break
                    
                if question.lower() == 'clear':
                    clear_screen()
                    continue
                
                if question.lower() == 'cache':
                    if self.answer_cache:
                        print("\n已缓存的问题:")
                        for i, q in enumerate(self.answer_cache, 1):
                            print(f"{i}. {q}")
                    else:
                        print("\n当前没有缓存的问题")
                    continue
                    
                # 处理实际问题
                start_time = time.time()
                answer = self.answer_question(question)
                end_time = time.time()
                
                # 美化输出
                print("\n" + "=" * 50)
                print("答案信息:")
                print("-" * 20)
                print(f"问题: {question}")
                print(f"\n简短回答: {answer['short_answer']}")
                if answer['context']:
                    print(f"\n相关内容:\n{answer['context']}")
                print(f"\n置信度: {answer['confidence']:.2f}")
                print(f"耗时: {end_time - start_time:.2f}秒")
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，正在退出...")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                print("\n抱歉，处理您的问题时出现错误，请重试。")

if __name__ == '__main__':
    # 设置环境变量
    os.environ['HF_HOME'] = 'E:\\huggingface\\transformers'
    
    try:
        embedding_path = os.path.join('embeddings', '基本医疗保险用药管理暂行办法_embedding.pkl')
        
        if not os.path.exists(embedding_path):
            print(f"错误：向量文件不存在: {embedding_path}")
            print("请先运行 generate_embeddings.py 生成向量文件")
            exit(1)
            
        qa_system = QASystem(embedding_path)
        qa_system.interactive_mode()
        
    except Exception as e:
        print(f"\n程序出现错误: {str(e)}")
        exit(1)