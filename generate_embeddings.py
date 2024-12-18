import os
import glob
import pickle
import warnings
import torch
from docx import Document
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

class DocumentEmbeddingGenerator:
    def __init__(self, docs_folder, embeddings_folder):
        self.docs_folder = docs_folder
        self.embeddings_folder = embeddings_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading tokenizer and model...")
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base",
            cache_dir='E:\\huggingface\\transformers',
            local_files_only=True
        )
        self.model = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base",
            cache_dir='E:\\huggingface\\transformers',
            local_files_only=True
        ).to(self.device)
        print("\n初始化完成。")

    def extract_text_from_docx(self, file_path):
        """从Word文档中提取文本"""
        try:
            doc = Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return text
        except Exception as e:
            print(f"提取文本时出错: {str(e)}")
            return None

    def generate_embeddings(self, text):
        """生成文本的向量表示"""
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        embeddings = []
        for para in paragraphs:
            inputs = self.tokenizer(
                para,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.pooler_output.cpu())
        
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return None

    def process_documents(self):
        """处理文档文件夹中的所有Word文档"""
        print("\n开始扫描文档目录...")
        
        print("\n目录中的所有文件:")
        for file in os.listdir(self.docs_folder):
            print(f"- {file}")
        
        doc_files = glob.glob(os.path.join(self.docs_folder, "*.docx"))
        
        print("\n找到的Word文件:")
        for file in doc_files:
            print(f"- {os.path.basename(file)}")
            output_name = f"{os.path.splitext(os.path.basename(file))[0]}_embedding.pkl"
            output_path = os.path.join(self.embeddings_folder, output_name)
            if os.path.exists(output_path):
                print(f"  (已有向量文件)")
            else:
                print(f"  (需要处理)")
        
        if not doc_files:
            print("\n错误：在 documents 文件夹中没有找到 .docx 文件！")
            return
        
        print(f"\n需要处理 {len(doc_files)} 个Word文档...")
        
        for file_path in doc_files:
            file_name = os.path.basename(file_path)
            output_name = f"{os.path.splitext(file_name)[0]}_embedding.pkl"
            output_path = os.path.join(self.embeddings_folder, output_name)
            
            if os.path.exists(output_path):
                print(f"\n跳过 {file_name} (向量文件已存在)")
                continue
            
            print(f"\n处理文档: {file_name}")
            try:
                # 提取文本
                content = self.extract_text_from_docx(file_path)
                if not content:
                    print(f"警告: {file_name} 内容提取失败")
                    continue
                
                print(f"- 成功提取文本，长度: {len(content)} 字符")
                
                # 生成向量
                embeddings = self.generate_embeddings(content)
                if embeddings is None:
                    print(f"警告: {file_name} 无法生成向量")
                    continue
                
                print(f"- 成功生成向量，维度: {embeddings.shape}")
                
                # 保存向量文件
                with open(output_path, 'wb') as f:
                    pickle.dump({
                        'embedding': embeddings,
                        'content': content
                    }, f)
                
                print(f"- 成功保存向量文件: {output_name}")
                
            except Exception as e:
                print(f"处理文档 {file_name} 时出错: {str(e)}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_folder = os.path.join(current_dir, 'documents')
    embeddings_folder = os.path.join(current_dir, 'embeddings')
    
    os.makedirs(docs_folder, exist_ok=True)
    os.makedirs(embeddings_folder, exist_ok=True)
    
    print(f"\n文档目录: {docs_folder}")
    print(f"向量目录: {embeddings_folder}")
    
    generator = DocumentEmbeddingGenerator(docs_folder, embeddings_folder)
    generator.process_documents()
    
    print("\n处理完成！") 