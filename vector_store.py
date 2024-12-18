import warnings
import multiprocessing
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import torch
import numpy as np
import os

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using.*TRANSFORMERS_CACHE.*")

def init():
    if __name__ == '__main__':
        multiprocessing.freeze_support()

# 设置环境变量
os.environ['HF_HOME'] = 'E:\\huggingface\\transformers'
os.environ['TRANSFORMERS_CACHE'] = 'E:\\huggingface\\transformers'

# 使用GPU加速（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型到GPU
print("Loading tokenizer and model...")
tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base",
    cache_dir='E:\\huggingface\\transformers',
    local_files_only=True  # 避免在线下载
)
model = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base",
    cache_dir='E:\\huggingface\\transformers',
    local_files_only=True  # 避免在线下载
).to(device)
print("Model loaded.")

def vectorize_document(content, batch_size=4):
    # 分段并批处理
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    embeddings = []
    
    # 批量处理
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            embedding = model(**inputs).pooler_output
            embeddings.append(embedding.cpu())
    
    if embeddings:
        document_embedding = torch.mean(torch.cat(embeddings), dim=0, keepdim=True)
    else:
        document_embedding = torch.zeros((1, model.config.hidden_size))
    
    return document_embedding

def store_vectors(embedding):
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # 如果GPU可用，使用GPU版本的FAISS
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embedding.numpy())
    return index

# Example usage for inference
inputs = tokenizer("Example text", return_tensors="pt", truncation=True, padding=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)