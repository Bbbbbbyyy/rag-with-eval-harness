# 本地化 RAG 问答与两级评测系统

基于 Ollama + LangChain + ChromaDB 搭建的本地 RAG 系统，支持 PDF 上传、向量检索、多轮对话，并内置两级评测 Harness。

## 功能

- **本地 RAG 问答**：上传 PDF，基于检索到的原文回答问题
- **检索溯源**：每次回答可展开查看 AI 参考的原文片段
- **多轮对话记忆**：支持追问，系统能理解代词
- **两级评测 Harness**：
  - 第一级：关键词匹配，快速判断回答是否包含预期关键词
  - 第二级：关键词失败时，调用 LLM-as-Judge 做语义判断
  - 自动输出通过率、关键词通过数、LLM 裁判通过数

## 技术栈

- **模型服务**：Ollama（llama3.2 + mxbai-embed-large）
- **框架**：LangChain、LangChain-Ollama
- **向量库**：ChromaDB
- **界面**：Streamlit
- **语言**：Python 3.11

## 运行步骤

### 1. 安装 Ollama 并拉取模型

    ollama pull llama3.2
    ollama pull mxbai-embed-large

### 2. 安装依赖

    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### 3. 启动问答界面

    streamlit run app.py

浏览器打开 http://localhost:8501

### 4. 运行评测 Harness

    python eval.py

输出示例：

    评测完成！通过率: 100.0% (12/12)
      其中关键词通过: 8，LLM裁判通过: 4

## 项目结构

    chatpdf-rag-deepseek-r1/
    ├── app.py              # Streamlit 界面
    ├── rag.py              # RAG 核心逻辑
    ├── eval.py             # 两级评测 Harness
    ├── data/pdfs/          # PDF 存放目录
    ├── chroma_db/          # 向量库（自动生成）
    └── requirements.txt    # 依赖清单

## 已知不足与下一步计划

- 评测集目前 12 条，计划扩展到 50 条以上，引入公开数据集
- LLM 裁判未做人工标注校准，计划计算 TPR/TNR
- 未跑 baseline 对比（纯 LLM vs RAG）
- 未做缓存、重试、并发优化

## 作者

鲍红颖 | 西北农林科技大学人工智能硕士（在读）
