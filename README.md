# DeepScholar-Agent

基于 **LangGraph** 构建的多智能体深度研究系统，面向 AI 技术、学术论文与开源项目调研。

系统能够将复杂研究问题自动拆分为多个 ResearchTask，并行检索 **Web、arXiv、GitHub** 等数据源，通过混合检索筛选相关内容、抽取结构化 Evidence，并利用 **Critic + Replanner** 动态补充研究缺口。在报告生成阶段，通过 **Claim Generator + Citation Verifier** 对事实与引用进行校验，最终生成带有可追溯引用的结构化研究报告。

## ✨ 主要特性

- **多智能体研究流程**：Planner、Critic、Replanner、Claim Generator、Citation Verifier、Report Writer
- **并行任务执行**：基于 LangGraph `Send` 实现 ResearchTask 级 Fan-out / Fan-in
- **多源检索**：统一接入 Web、arXiv、GitHub
- **混合检索**：BGE Dense Retrieval + BM25 + RRF 融合排序
- **Token Budget**：根据模型上下文预算动态筛选高相关 Chunk
- **动态重规划**：Critic 检测研究覆盖情况，Replanner 针对知识缺口补充任务
- **引用校验**：Evidence → Claim → Citation Verification，减少事实与引用错配
- **容错机制**：Tool Retry、Source / Task Failure Isolation
- **断点恢复**：基于 LangGraph SQLite Checkpoint 保存和恢复研究状态

## 🏗️ 系统架构

```text
                         User Query
                              │
                              ▼
                           Planner
                              │
                    ┌─────────┴─────────┐
                    │  Research Tasks   │
                    └─────────┬─────────┘
                              │
                    LangGraph Send
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        ResearchTask     ResearchTask     ResearchTask
             │                │                │
        Search/Fetch      Search/Fetch      Search/Fetch
             │                │                │
        Chunk/Retrieve    Chunk/Retrieve    Chunk/Retrieve
             │                │                │
        Evidence Extract  Evidence Extract  Evidence Extract
             └────────────────┼────────────────┘
                              │
                              ▼
                         Evidence Pool
                              │
                              ▼
                            Critic
                              │
                    ┌─────────┴─────────┐
                    │                   │
                Sufficient         Insufficient
                    │                   │
                    │               Replanner
                    │                   │
                    │            New Research Tasks
                    │                   │
                    └───────────◄───────┘
                    │
                    ▼
              Evidence Processor
                    │
                    ▼
               Claim Generator
                    │
                    ▼
              Citation Verifier
                    │
                    ▼
                Report Writer
                    │
                    ▼
                Final Report
```

## 🔍 核心流程

### 1. 研究规划

Planner 根据用户问题生成 3～5 个相互独立、互补的研究任务，并根据研究内容选择对应数据源：

- `PAPER`：论文、模型架构、训练方法、Benchmark、实验结果
- `WEB`：技术文档、新闻、项目公告、生态信息
- `GITHUB`：开源实现、代码仓库、README

### 2. 多源检索

不同数据源通过统一的 Search / Fetch 接口接入系统：

```text
Web     → Tavily Search → Web Fetcher
arXiv   → arXiv API     → Paper Fetcher
GitHub  → GitHub Search → Repository Fetcher
```

不同来源最终统一转换为 `SourceDocument`，后续共用同一套处理流程。

### 3. Chunk 检索与 Evidence 抽取

长文档首先进行递归 Chunking，再通过：

```text
BM25
  +
BGE Dense Retrieval
  ↓
 RRF
  ↓
Top-K Chunks
```

筛选与当前 ResearchTask 最相关的内容。

Context Builder 根据 Token Budget 控制送入 LLM 的上下文长度，随后 Evidence Extractor 从相关 Chunk 中抽取结构化 Evidence。

### 4. Critic 与动态 Replan

Critic 根据**用户原始研究需求**判断当前 Evidence 是否已经覆盖核心研究维度。

当存在影响最终回答的 Blocking Gap 时：

```text
Critic
  ↓
Knowledge Gaps
  ↓
Replanner
  ↓
New Research Tasks
  ↓
Research
```

Replanner 只生成解决知识缺口所需的最少新任务，同时通过 `max_replans` 防止无限重规划。

### 5. Claim 与引用验证

为了降低多篇资料混合时的错误归纳和引用错配，报告生成前增加事实验证流程：

```text
Evidence
   ↓
Claim Generator
   ↓
Atomic Claims
   ↓
Citation Verifier
   ↓
Verified Claims
   ↓
Report Writer
```

Citation Verifier 检查 Claim 与 Evidence 在主体、方法、指标、数值、比较关系和适用条件等方面是否一致，只保留有证据支撑的 Claim。

### 6. 容错与状态恢复

针对长流程研究任务，实现：

- Tool Retry 与指数退避
- Source Failure Isolation
- ResearchTask Failure Isolation
- SQLite Checkpoint
- 基于 `thread_id` 的断点续跑

单个数据源或 ResearchTask 失败不会直接导致整个研究流程终止。

## 📁 项目结构

```text
DeepScholar-Agent/
├── src/
│   └── deepscholar/
│       ├── agents/          # Planner / Critic / Replanner 等
│       ├── graph/           # LangGraph 工作流与 State
│       ├── tools/           # Web / arXiv / GitHub 工具
│       ├── services/        # ResearchWorker、检索、ContextBuilder 等
│       ├── models/          # Pydantic 数据模型
│       ├── prompts/         # Agent Prompts
│       ├── llm/
│       └── utils/
├── run_research.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/LaityLu/DeepScholar-Agent.git
cd DeepScholar-Agent
```

### 2. 安装依赖

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境：

```bash
uv sync
```

### 3. 配置环境变量
在项目根目录下创建 `.env` 文件：
```
# 数据源
TAVILY_API_KEY=your_tavily_api_key
GITHUB_TOKEN=xxx

# LLM 配置，项目使用 vllm 本地部署 Qwen3.5-4B
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=http://xxx/v1
LLM_MODEL=xxx
LLM_MAX_CONTEXT_TOKENS=xxx
LLM_MAX_OUTPUT_TOKENS=xxx

# 模型配置
LLM_TOKENIZER_PATH=xxx
EMBEDDING_MODEL_PATH=xxx
```

### 4. 运行

```bash
# 首次运行任务，系统会创建唯一的 Research Thread
uv run run_research.py \
"Research the latest progress of multimodal GUI agents from 2025 to 2026."
# 断点恢复
uv run scripts/run_research.py \
  --resume thread_id
```


## 🛠️ 技术栈

- **Agent / Workflow**：LangGraph
- **LLM**：Qwen + vLLM
- **Embedding**：BGE
- **Retrieval**：BM25 + Dense Retrieval + RRF
- **Web Search**：Tavily
- **Paper Search**：arXiv API
- **Open Source Search**：GitHub
- **Checkpoint**：SQLite
- **Dependency Management**：uv

