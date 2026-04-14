# NoiseFilter-RAG Reproduction Guide | 复现指南

This guide is bilingual (English + 中文) with clickable anchors.
本文为中英双语文档，并提供可点击锚点跳转。

## Table of Contents | 目录

- [1. Goal | 目标](#goal)
- [2. Implemented Scope | 已实现内容](#implemented-scope)
- [3. Reproduction Tracks | 复现路径](#reproduction-tracks)
- [4. Environment Setup | 环境配置](#environment-setup)
- [4.1 Local Full-Stack Bring-up | 本地全链路启动](#local-fullstack)
- [5. Dataset Workflow | 数据集流程](#dataset-workflow)
- [6. Query and Evaluation | 查询与评估](#query-and-evaluation)
- [7. Synthetic Noise Benchmark | 合成噪声基准](#synthetic-noise-benchmark)
- [8. Deliverables | 交付物](#deliverables)
- [9. Repo-Level Improvements | 仓库改进建议](#repo-level-improvements)
- [10. Remaining Actions | 剩余操作](#remaining-actions)
- [11. Commands Checklist | 命令清单](#commands-checklist)
- [12. File Map | 文件映射](#file-map)

---

## 1. Goal | 目标
<a id="goal"></a>

Build this LightRAG fork into a polished, reproducible NoiseFilter-RAG project.
将当前 LightRAG fork 打造成可稳定复现、可展示的 NoiseFilter-RAG 工程。

Core targets | 核心目标:
- confidence scoring for extracted relation edges | 为抽取关系边打置信分
- noise-aware retrieval with soft and hard filtering | 支持软/硬过滤的噪声感知检索
- synthetic noise benchmark with exportable reports | 提供可导出的合成噪声基准结果
- one operational guide for all manual steps | 用一份文档覆盖所有手工步骤

## 2. Implemented Scope | 已实现内容
<a id="implemented-scope"></a>

Implemented modules | 已实现模块:
- lightrag/noisefilter/confidence.py (ConfidenceScoringEngine)
- lightrag/noisefilter/retriever.py (NoiseAwareRetriever)
- lightrag/noisefilter/benchmark.py (NoiseInjector, evaluate_filter)
- lightrag/noisefilter/experiment.py (JSON/CSV/Markdown export helpers)
- lightrag/noisefilter/reproduction.py (formal dataset reproduction helpers)
- reproduce/noisefilter_experiment.py (synthetic benchmark CLI)
- reproduce/run_formal_insert.py and reproduce/run_formal_query.py (generic formal runners)
- reproduce/run_baseline_insert.py / run_noisefilter_insert.py (fixed-variant insert runners)
- reproduce/run_baseline_query.py / run_noisefilter_query.py (fixed-variant query runners)
- examples/noisefilter_demo.py (local end-to-end demo)
- tests/test_noise_filter.py, tests/test_noise_filter_experiment.py, and tests/test_noise_filter_formal_reproduction.py

Integrated behaviors | 已接入流程:
- LightRAG(enable_noise_filter=True, noise_filter_config=...)
- post-insert confidence scoring after relation merge
- noise-aware relation ranking at query time
- query_data() returns conf_score and related metadata

## 3. Reproduction Tracks | 复现路径
<a id="reproduction-tracks"></a>

### Track A: Quick Local Validation | 本地快速验证

Run this first. | 建议先跑这一条路径。
```bash
./scripts/test.sh tests/test_noise_filter.py
./scripts/test.sh tests/test_noise_filter_experiment.py
python3 examples/noisefilter_demo.py
python3 reproduce/noisefilter_experiment.py
```

Expected outputs | 预期输出:
- tests pass
- demo prints confidence scores and soft/hard retrieval differences
- experiment writes JSON/CSV/Markdown under reproduce/results/noisefilter/

### Track B: Full Pipeline Reproduction | 完整流程复现

Recommended sequence | 推荐顺序:
1. create Python environment and install dependencies | 创建 Python 环境并安装依赖
2. run setup wizard (`make env-base`, `make env-storage`, `make env-server`) | 运行配置向导
3. bring up required middleware containers | 启动所需中间件容器
4. start LightRAG API service and verify `/docs` | 启动 LightRAG API 并验证 `/docs`
5. optional: start WebUI and bind API endpoint | 可选：启动 WebUI 并绑定 API 地址
6. download dataset | 下载数据集
7. extract unique contexts | 抽取去重上下文
8. build baseline index | 构建基线索引
9. build noisefilter index | 构建 NoiseFilter 索引
10. run query generation and execution | 执行问题生成与查询
11. run answer-quality evaluation | 执行答案质量评估
12. collect synthetic benchmark results | 汇总合成噪声结果
13. produce final tables and figures | 产出最终图表

## 4. Environment Setup | 环境配置
<a id="environment-setup"></a>

### Python Environment | Python 环境
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[test,evaluation]"
```

If API workflow is also needed | 若还要跑 API 流程:

```bash
pip install -e ".[api,test,evaluation]"
```

LightRAG setup (recommended) | 推荐配置命令:

```bash
make env-base
```

Formal reproduction env vars | 正式复现实验建议环境变量:

```bash
export LLM_MODEL=gpt-4o-mini
export LLM_BINDING_API_KEY=your_llm_api_key
export LLM_BINDING_HOST=https://api.openai.com/v1

export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_BINDING_API_KEY=your_embedding_api_key
export EMBEDDING_BINDING_HOST=https://api.openai.com/v1

export EMBEDDING_DIM=1536
export MAX_EMBED_TOKENS=8192
```

RAGAS evaluation env bridge | RAGAS 评测环境变量桥接:

`lightrag/evaluation/eval_rag_quality.py` expects `EVAL_*` variables.
`eval_rag_quality.py` 期望读取 `EVAL_*` 变量。

If your `.env` only defines `LLM_*` / `EMBEDDING_*`, bridge them before evaluation:
如果你的 `.env` 只有 `LLM_*` / `EMBEDDING_*`，请在评测前桥接：

```bash
export EVAL_LLM_BINDING_API_KEY="$LLM_BINDING_API_KEY"
export EVAL_LLM_BINDING_HOST="$LLM_BINDING_HOST"
export EVAL_LLM_MODEL="$LLM_MODEL"

export EVAL_EMBEDDING_BINDING_API_KEY="$EMBEDDING_BINDING_API_KEY"
export EVAL_EMBEDDING_BINDING_HOST="$EMBEDDING_BINDING_HOST"
export EVAL_EMBEDDING_MODEL="$EMBEDDING_MODEL"
```

NoiseFilter switch example | NoiseFilter 开关示例:

```python
rag = LightRAG(
    working_dir="./rag_storage/noisefilter",
    llm_model_func=your_llm_func,
    embedding_func=your_embedding_func,
    enable_noise_filter=True,
    noise_filter_config={
        "w_freq": 0.5,
        "w_cons": 0.3,
        "w_sem": 0.2,
        "conf_threshold": 0.3,
        "soft_mode": True,
    },
)
```

### 4.1 Local Full-Stack Bring-up (Middleware + API + WebUI) | 本地全链路启动（中间件 + API + WebUI）
<a id="local-fullstack"></a>

Goal | 目标:
- bring up a runnable local stack before dataset ingestion and evaluation
- 在开始数据构建和评估前，先拉起可运行的本地完整链路

Step A: Setup Wizard (recommended) | 第 A 步：配置向导（推荐）
```bash
make env-base
make env-storage
make env-server
```

Step B: Start middleware services | 第 B 步：启动中间件服务
Use the wizard-generated compose file when available.
优先使用向导生成的 compose 文件。

```bash
docker compose -f docker-compose.final.yml up -d
docker compose -f docker-compose.final.yml ps
```

If `docker-compose.final.yml` is not generated, fallback to default compose.
若未生成 `docker-compose.final.yml`，可回退到默认 compose。

```bash
docker compose -f docker-compose.yml up -d
docker compose -f docker-compose.yml ps
```

Step C: Start API service | 第 C 步：启动 API 服务
```bash
lightrag-server
# or
uvicorn lightrag.api.lightrag_server:app --host 0.0.0.0 --port 9621 --reload
```

Step D: Verify API service | 第 D 步：验证 API 服务
- Open `http://localhost:9621/docs` in browser and confirm routes are visible.
- 浏览器打开 `http://localhost:9621/docs`，确认接口已加载。

Optional quick check | 可选快速检查:
```bash
curl -sS http://localhost:9621/docs > /tmp/lightrag_api_docs.html && echo "API docs reachable"
```

Step E: Start WebUI (optional but recommended) | 第 E 步：启动 WebUI（可选但推荐）
```bash
cd lightrag_webui
bun install
bun run dev
```

- Default dev URL is usually `http://localhost:5173`.
- 默认开发地址通常为 `http://localhost:5173`。

Step F: End-to-end smoke test | 第 F 步：端到端冒烟验证
1. run `examples/noisefilter_demo.py` to verify core NoiseFilter behavior.
2. start API and run one query request from WebUI or API client.
3. confirm retrieval includes confidence-related metadata.

1. 运行 `examples/noisefilter_demo.py` 验证核心 NoiseFilter 行为。
2. 启动 API 后，从 WebUI 或 API 客户端发送一次查询。
3. 确认返回中包含置信度相关元数据。

## 5. Dataset Workflow | 数据集流程
<a id="dataset-workflow"></a>

Official source noted upstream | 上游说明数据源:
- TommyChien/UltraDomain (Hugging Face)

Suggested layout | 建议目录结构:

```text
datasets/
├── raw/
├── unique_contexts/
├── questions/
└── results/
```

Step 1 | 第一步: extract unique contexts

```bash
python3 reproduce/Step_0.py -i ./datasets/raw -o ./datasets/unique_contexts
```

Step 2 | 第二步: build two indexes
- baseline: enable_noise_filter=False
- noisefilter: enable_noise_filter=True

Recommended names | 推荐目录名:
- rag_storage/agriculture_baseline
- rag_storage/agriculture_noisefilter

Recommended formal runners | 推荐正式 runner:

```bash
# baseline insert
python3 reproduce/run_baseline_insert.py --dataset agriculture

# noisefilter insert
python3 reproduce/run_noisefilter_insert.py --dataset agriculture
```

Generic runner equivalents | 通用 runner 等价命令:

```bash
python3 reproduce/run_formal_insert.py --dataset agriculture --variant baseline
python3 reproduce/run_formal_insert.py --dataset agriculture --variant noisefilter
```

Default path convention | 默认目录约定:

```text
datasets/unique_contexts/{dataset}_unique_contexts.json
datasets/questions/{dataset}_questions.txt
rag_storage/formal_runs/{dataset}/{variant}/
reproduce/results/formal/{dataset}/{variant}/
```

## 6. Query and Evaluation | 查询与评估
<a id="query-and-evaluation"></a>

### Query Generation | 问题生成

```bash
python3 reproduce/Step_2.py
```

### Query Execution | 查询执行

Run the same question set for baseline and noisefilter.
用同一套问题分别跑 baseline 与 noisefilter。

Recommended formal query commands | 推荐正式查询命令:

```bash
# baseline query
python3 reproduce/run_baseline_query.py --dataset agriculture --mode hybrid

# noisefilter query
python3 reproduce/run_noisefilter_query.py --dataset agriculture --mode hybrid
```

Generic runner equivalents | 通用 runner 等价命令:

```bash
python3 reproduce/run_formal_query.py --dataset agriculture --variant baseline --mode hybrid
python3 reproduce/run_formal_query.py --dataset agriculture --variant noisefilter --mode hybrid
```

Formal output layout | 正式输出目录:

```text
reproduce/results/formal/
└── agriculture/
    ├── baseline/
    │   ├── hybrid_results.json
    │   └── hybrid_errors.json
    └── noisefilter/
        ├── hybrid_results.json
        └── hybrid_errors.json
```

Suggested result layout | 结果示例:

```text
datasets/results/
├── agriculture_baseline_result.json
├── agriculture_noisefilter_result.json
├── cs_baseline_result.json
├── cs_noisefilter_result.json
└── ...
```

### Answer Quality Evaluation | 答案质量评估

Option 1: reproduce/batch_eval.py (pairwise)
选项 1: 使用 reproduce/batch_eval.py 做 pairwise 评估

Option 2: RAGAS evaluator
选项 2: 使用 RAGAS 评估器

Recommended standard benchmark first | 推荐先跑标准样例基准:

1. index `lightrag/evaluation/sample_documents/*.md`
2. run `sample_dataset.json` against the API
3. verify average RAGAS score is around or above `0.80`

1. 先把 `lightrag/evaluation/sample_documents/*.md` 建库。
2. 再用 `sample_dataset.json` 对 API 做评测。
3. 观察平均 RAGAS 分数是否达到 `0.80` 左右或更高。

```bash
python3 lightrag/evaluation/eval_rag_quality.py \
  --dataset lightrag/evaluation/sample_dataset.json \
  --ragendpoint http://localhost:9621
```

Compare metrics | 建议对比指标:
- Faithfulness
- Answer Relevance
- Context Recall
- Context Precision

## 7. Synthetic Noise Benchmark | 合成噪声基准
<a id="synthetic-noise-benchmark"></a>

Fast reproducible quantitative artifact.
最快可复现的量化产出。

```bash
python3 reproduce/noisefilter_experiment.py
```

Custom sweep | 自定义扫描:

```bash
python3 reproduce/noisefilter_experiment.py \
  --noise-ratios 0.1,0.2,0.3,0.4 \
  --thresholds 0.1,0.2,0.3,0.4,0.5 \
  --top-k 5 \
  --injectors random,contradictory
```

Outputs | 输出:
- JSON (machine-readable metrics)
- CSV (spreadsheet-ready rows)
- Markdown (report-ready summary)

Key metrics | 关键指标:
- filter_precision
- filter_recall
- filter_f1
- retrieval_noise_rate_at_k
- avg_returned_edges

## 8. Deliverables | 交付物
<a id="deliverables"></a>

- one baseline result directory per domain
- one noisefilter result directory per domain
- one synthetic benchmark bundle
- one RAGAS comparison table
- one ablation table (full/no-freq/no-consistency/no-semantic/hard/soft)
- one final README results section

## 9. Repo-Level Improvements | 仓库改进建议
<a id="repo-level-improvements"></a>

- extend the dedicated baseline/noisefilter runners with batch sweeps and report aggregation
- add JSON-to-Markdown table generation
- add plotting scripts (threshold vs F1, threshold vs noise rate, baseline vs soft vs hard)
- add domain-wise final sheet under reproduce/results/final/

## 10. Remaining Actions | 剩余操作
<a id="remaining-actions"></a>

### Mandatory | 必做
1. Create and activate Python environment. | 创建并激活 Python 环境。
2. Install editable dependencies (`.[api,test,evaluation]`). | 安装可编辑依赖（含 API）。
3. Configure `make env-base`, `make env-storage`, `make env-server`. | 完成基础、存储、服务配置。
4. Bring up middleware containers and verify status. | 启动中间件容器并确认状态。
5. Start LightRAG API and verify `/docs`. | 启动 LightRAG API 并验证 `/docs`。
6. (Optional) Start WebUI and connect API endpoint. | （可选）启动 WebUI 并连接 API。
7. Download UltraDomain to `datasets/raw/`. | 下载数据到 `datasets/raw/`。
8. Run `reproduce/run_baseline_insert.py` and `reproduce/run_noisefilter_insert.py`. | 运行 baseline/noisefilter 正式建库脚本。
9. Run `reproduce/run_baseline_query.py` and `reproduce/run_noisefilter_query.py`, then evaluate outputs. | 运行正式查询脚本并评估结果。
10. Archive report-ready outputs. | 归档可汇报结果。

### Strongly Recommended | 强烈建议
1. Freeze directory naming before long runs. | 长跑前固定目录命名。
2. Keep baseline/noisefilter outputs separated. | baseline/noisefilter 输出严格分离。
3. Record model names/endpoints/dates. | 记录模型名、接口和日期。
4. Keep synthetic benchmark bundle. | 保存合成噪声结果包。
5. Export final Markdown summary. | 导出最终 Markdown 总结。

### Nice to Have | 可选加分
1. Add baseline noisy-neighbor failure cases. | 增加基线噪声邻居失败案例。
2. Capture result/context screenshots. | 截图记录结果与上下文分析。
3. Add bilingual short technical report section. | 增加中英双语技术报告段落。

## 11. Commands Checklist | 命令清单
<a id="commands-checklist"></a>

```bash
# 1) environment | 环境准备
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[api,test,evaluation]"

# 2) setup wizard | 交互式配置
make env-base
make env-storage
make env-server

# 3) middleware up (preferred) | 启动中间件（推荐）
docker compose -f docker-compose.final.yml up -d
docker compose -f docker-compose.final.yml ps

# 4) fallback compose | 回退默认 compose（如未生成 final 文件）
docker compose -f docker-compose.yml up -d
docker compose -f docker-compose.yml ps

# 5) API server | 启动 API
lightrag-server
# or
uvicorn lightrag.api.lightrag_server:app --host 0.0.0.0 --port 9621 --reload

# 6) API quick verification | API 快速验证
curl -sS http://localhost:9621/docs > /tmp/lightrag_api_docs.html && echo "API docs reachable"

# 7) local verification | 本地功能验证
./scripts/test.sh tests/test_noise_filter.py
./scripts/test.sh tests/test_noise_filter_experiment.py
python3 examples/noisefilter_demo.py
python3 reproduce/noisefilter_experiment.py

# 8) dataset prep | 数据预处理
python3 reproduce/Step_0.py -i ./datasets/raw -o ./datasets/unique_contexts

# 9) formal insert | 正式建库
python3 reproduce/run_baseline_insert.py --dataset agriculture
python3 reproduce/run_noisefilter_insert.py --dataset agriculture

# 10) formal query | 正式查询
python3 reproduce/run_baseline_query.py --dataset agriculture --mode hybrid
python3 reproduce/run_noisefilter_query.py --dataset agriculture --mode hybrid

# 11) bridge eval env vars when needed | 必要时桥接评测环境变量
export EVAL_LLM_BINDING_API_KEY="$LLM_BINDING_API_KEY"
export EVAL_LLM_BINDING_HOST="$LLM_BINDING_HOST"
export EVAL_LLM_MODEL="$LLM_MODEL"
export EVAL_EMBEDDING_BINDING_API_KEY="$EMBEDDING_BINDING_API_KEY"
export EVAL_EMBEDDING_BINDING_HOST="$EMBEDDING_BINDING_HOST"
export EVAL_EMBEDDING_MODEL="$EMBEDDING_MODEL"

# 12) query + eval | 查询与评估
python3 lightrag/evaluation/eval_rag_quality.py --ragendpoint http://localhost:9621

# 13) optional webui | 可选启动 WebUI
cd lightrag_webui && bun install && bun run dev
```

## 12. File Map | 文件映射
<a id="file-map"></a>

- implementation: lightrag/noisefilter/
- formal reproduction core: lightrag/noisefilter/reproduction.py
- demo: examples/noisefilter_demo.py
- synthetic experiment: reproduce/noisefilter_experiment.py
- formal runners: reproduce/run_formal_insert.py, reproduce/run_formal_query.py, reproduce/run_baseline_insert.py, reproduce/run_noisefilter_insert.py, reproduce/run_baseline_query.py, reproduce/run_noisefilter_query.py
- tests: tests/test_noise_filter.py, tests/test_noise_filter_experiment.py, tests/test_noise_filter_formal_reproduction.py
- this guide: docs/NoiseFilterRAG_Reproduction_Guide.md
