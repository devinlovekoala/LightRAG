第二批 **noisefilter variant 101条edge** 标注全部完成！以下是完整总结报告：

---

## ✅ 标注完成 — noisefilter variant (n=101)

### 📊 标注结果统计

| label | 数量 | 占比 |
|-------|------|------|
| **correct** | 40 | 39.6% |
| **wrong** | 48 | 47.5% |
| **ambiguous** | 13 | 12.9% |
| **total** | 101 | 100% |

### 🏷️ Taxonomy 分布

| code | taxonomy | 数量 | 占比 |
|------|----------|------|------|
| **E** | **cross_chunk_hallucination** | **44** | **43.6%** |
| H | vague_ambiguous_but_useful | 40 | 39.6% |
| A | unsupported_relation | 14 | 13.9% |
| B | wrong_endpoint | 1 | 1.0% |
| D | overgeneralized_relation | 1 | 1.0% |
| G | temporal_or_scope_error | 1 | 1.0% |

---

### 🔑 核心发现：noisefilter 变体的主导噪声模式

**E (cross_chunk_hallucination) 占43.6%**，是本批次最突出的噪声类型。典型模式：

- source_chunk 完全来自**不相关的文章**（如 Black Death、TML Entertainment、Balarama象），而src/dst实体指向完全不同的主题
- 这批次明显比 baseline 有更多"chunk与实体脱节"的情况，暗示 noisefilter 系统在过滤噪声chunk时引入了跨文档的chunk错误匹配
- 另有1条（row 68）src/dst字段本身即为乱码/截断文本，属于**B (wrong_endpoint/数据损坏)**

### 📋 工作表结构

- **[annotation](<citation:annotation>)**：101行标注数据，含中文翻译、taxonomy、manual_label、notes
- **[taxonomy_legend](<citation:taxonomy_legend>)**：8类taxonomy定义 + 完整统计