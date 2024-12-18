import torch
import logging
from document_parser import extract_text_from_docx
from vector_store import vectorize_document, store_vectors
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_embeddings(document_path, output_dir='embeddings'):
    """将文档转换为向量并保存"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        doc_name = os.path.basename(document_path).split('.')[0]
        embedding_path = os.path.join(output_dir, f'{doc_name}_embedding.pkl')
        
        # 检查是否已存在
        if os.path.exists(embedding_path):
            logger.info(f"向量文件已存在: {embedding_path}")
            return embedding_path
            
        # 解析文档
        logger.info(f"开始处理文档: {document_path}")
        document_content = extract_text_from_docx(document_path)
        
        # 向量化
        logger.info("开始向量化...")
        document_embedding = vectorize_document(document_content, batch_size=8)
        
        # 保存向量和文档内容
        data = {
            'embedding': document_embedding,
            'content': document_content
        }
        with open(embedding_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"向量保存完成: {embedding_path}")
        return embedding_path
        
    except Exception as e:
        logger.error(f"向量生成过程出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # 指定文档路径
    doc_path = 'docs/基本医疗保险用药管理暂行办法.docx'
    save_embeddings(doc_path) 