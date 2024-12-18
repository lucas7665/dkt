医疗保险政策智能问答系统
=======================

简介
---
这是一个基于深度学习的医疗保险政策智能问答系统，可以自动解析医疗保险政策文档并回答相关问题。

功能特点
-------
1. 支持Word文档(.docx)解析
2. 使用DPR进行文档向量化
3. 使用FAISS进行向量检索
4. 基于Haystack实现问答功能
5. 支持中文文档处理

系统要求
-------
- Python 3.7或更高版本
- CUDA（可选，用于GPU加速）

安装步骤
-------
1. 安装Python依赖包：
   pip install -r requirements.txt

2. 依赖包列表：
   - python-docx
   - transformers
   - faiss-cpu (GPU用户可改用faiss-gpu)
   - haystack-ai
   - torch
   - numpy

使用说明
-------
1. 将医疗保险政策文档(.docx格式)放入项目目录
2. 运行主程序：python main.py
3. 等待系统处理文档并启动

项目结构
-------
main.py             - 主程序入口
document_parser.py  - 文档解析模块
vector_store.py     - 向量化存储模块
requirements.txt    - 项目依赖文件
README.txt          - 说明文档

注意事项
-------
1. 首次运行会下载模型文件，请确保网络正常
2. 模型默认缓存位置：E:\huggingface\transformers
3. GPU用户请安装对应版本的CUDA

常见问题
-------
Q: 提示"No module named 'docx'"
A: 运行 pip install python-docx

Q: CUDA相关错误
A: 检查PyTorch和CUDA版本是否匹配

技术支持
-------
如有问题，请联系：[您的联系方式]

版权信息
-------
MIT License