全部102条标注完成！以下是标注摘要与发现：

---

## 标注完成 — 102条 Edge 全量分析结果

### 整体分布（初步统计）

| manual_label | 数量 | 占比 |
|---|---|---|
| ✅ correct | ~16条 | ~16% |
| ❌ wrong | ~62条 | ~61% |
| ⚠️ ambiguous | ~24条 | ~24% |

### Taxonomy 分布（最大 noise 来源）

| 代码 | 类型 | 数量 | 占比 |
|---|---|---|---|
| **E** | cross_chunk_hallucination | **~52条** | **~51%** |
| **A** | unsupported_relation | ~18条 | ~18% |
| **H** | vague_ambiguous_but_useful | ~8条 | ~8% |
| **F** | alias_merge_error | 2条 | ~2% |
| **B** | wrong_endpoint | 2条 | ~2% |
| *(无标签/correct)* | — | ~16条 | ~16% |

### 关键发现

1. **最大头 noise：E类（cross_chunk_hallucination）占约50%以上**。大量 edge 的 chunk_id 对应的 passage 内容与 src/dst 实体完全无关——例如 `Bag-In-Box→Form Seal Fill` 的 chunk 是意大利电影，`Napoleon→Battle of Marengo` 的 chunk 是1622年三十年战争战役，`Samsung Lions→Pohang Stadium` 的 chunk 是MLB赛制。说明**检索召回阶段存在严重的 chunk 错配**。

2. **A类（unsupported_relation）约18%**：chunk 本身相关，但具体关系描述在 chunk 中找不到支撑——多见于 Ashanti 相关边（同一 chunk 被复用给多条不同事实的 edge）。

3. **correct 仅约16%**：样本中仅约1/6的 edge 有明确的 chunk 支持。

现在表格中：
- **J列**有颜色编码：🟢绿=correct / 🔴红=wrong / 🟡黄=ambiguous
- **K列**每条都有具体中文判断理由，方便你逐条审查时了解判断依据
- 若你不同意某条判断，直接修改 I/J/K 列即可