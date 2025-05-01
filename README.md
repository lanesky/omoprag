# OMOP Concept Retriever with OpenAI and ChromaDB

## 项目文件概述

### 配置文件

#### `config_openai.py`
- 包含项目的所有配置参数
- 设置OpenAI API密钥和模型选择
- 配置数据文件路径（CONCEPT.csv）
- 配置ChromaDB存储路径和集合名称
- 负责环境变量加载和基本验证

### 索引构建

#### `vocabulary_indexer_openai_streaming.py`
- 改进版的索引构建器，支持大规模数据集
- 以数据块（chunk）方式处理CONCEPT.csv文件
- 支持通过命令行参数过滤特定领域（如Drug）的概念
- 根据领域动态设置ChromaDB路径
- 包含更强大的错误处理和重试逻辑
- 提供详细的进度和性能日志

### 概念检索

#### `concept_retriever_openai.py`
- 主要的概念检索工具
- 接受用户查询（支持中文、日文或英文）
- 使用OpenAI API将查询翻译成英文（如需要）
- 生成查询的嵌入向量
- 在ChromaDB中搜索相似概念
- 支持按领域（domain）和词汇（vocabulary）过滤结果
- 支持在全部数据（All）或特定领域数据库中搜索
- 格式化并显示检索结果

#### `concept_retriever_openai_quick.py`
- 优化版的概念检索工具
- 首先检索更多的候选概念，然后在Python中进行过滤
- 提供更快的检索速度和更好的结果质量
- 支持在全部数据（All）或特定领域数据库中搜索
- 增强的错误处理和用户反馈

### 工具函数

#### `translator_openai.py`
- 提供文本翻译功能
- 使用OpenAI API将医学术语和相关文本翻译成目标语言
- 专为医学术语的准确翻译而设计
- 包含错误处理和回退机制

## 使用流程

1. 配置 `config_openai.py` 中的参数
2. 使用 `vocabulary_indexer_openai_streaming.py` 构建概念索引（可选择特定领域）
3. 使用 `concept_retriever_openai.py` 或 `concept_retriever_openai_quick.py` 检索概念

## 示例

```bash
# 构建所有概念的索引（存储在All目录）
python vocabulary_indexer_openai_streaming.py

# 仅构建Drug领域概念的索引（存储在Drug目录）
python vocabulary_indexer_openai_streaming.py Drug

# 在特定领域数据库中检索概念
python concept_retriever_openai.py "高血压" Condition

# 在全部数据（All）数据库中检索特定领域的概念(优点是在所有数据中匹配语义以及domain，精度高。缺点是慢，一次查询都有可能在3分钟以上。)
python concept_retriever_openai.py "高血压" Condition --use-all-db

# 使用优化版检索工具（先用concept_name进行语义匹配，再在候选中过滤domain，所以速度快。缺点是如果使用`--use-all-db`，候选concept中包含不匹配的domain，导致匹配的domain没有在候选中出现，也就是结果会不准确）
python concept_retriever_openai_quick.py "高血压" Condition
```

## 数据准备与索引构建过程

### 1. 获取OMOP概念数据

1. 从 [ATHENA](https://athena.ohdsi.org/) 下载 OMOP Vocabulary数据
   - 注册并登录ATHENA网站
   - 下载最新版本的Vocabulary数据包
   - 解压缩数据包，获取CONCEPT.csv文件

2. （可选）将CONCEPT.csv导入到PostgreSQL数据库中进行处理
   ```sql
   CREATE TABLE concept (
     concept_id INTEGER PRIMARY KEY,
     concept_name VARCHAR(255),
     domain_id VARCHAR(20),
     vocabulary_id VARCHAR(20),
     concept_class_id VARCHAR(20),
     standard_concept VARCHAR(1),
     concept_code VARCHAR(50),
     valid_start_date DATE,
     valid_end_date DATE,
     invalid_reason VARCHAR(1)
   );
   
   COPY concept FROM '/path/to/CONCEPT.csv' DELIMITER E'\t' CSV HEADER;
   ```

### 2. 获取CPT4数据（可选）

1. 申请CPT4的License Key
   - 访问 [AMA CPT网站](https://www.ama-assn.org/practice-management/cpt)
   - 按照指导申请CPT4的License

2. 使用License Key下载CPT4数据

3. 将CPT4数据合并到CONCEPT.csv文件中

### 3. 准备索引数据

1. 从完整的CONCEPT.csv中过滤出标准概念（Standard Concept）
   - 可以使用SQL或者pandas进行过滤
   - 保存到`csv_for_indexing/CONCEPT.csv`文件中

2. 配置`config_openai.py`文件
   - 设置OpenAI API密钥
   - 配置数据文件路径和ChromaDB存储路径

### 4. 构建索引

1. 使用`vocabulary_indexer_openai_streaming.py`构建全部概念的索引
   ```bash
   python vocabulary_indexer_openai_streaming.py
   ```

2. 或者按领域构建特定概念的索引
   ```bash
   python vocabulary_indexer_openai_streaming.py Drug
   python vocabulary_indexer_openai_streaming.py Condition
   python vocabulary_indexer_openai_streaming.py Procedure
   # ... 等等
   ```

构建索引的过程可能需要一定时间，取决于概念数据的规模和您的OpenAI API限制。索引完成后，您就可以使用概念检索工具进行查询了。
